export type Precontext = Record<string, unknown>;

const TAG_RE = (tag: string) => new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`, "g");

/** Pull `<think>`/`<precontext>` blocks out of content; returns the rest as `text`. */
export function stripSideChannels(content: string): {
  text: string;
  reasoning?: string;
  precontext?: Precontext[];
} {
  let text = content;
  const thinks: string[] = [];
  text = text.replace(TAG_RE("think"), (_m, inner: string) => {
    thinks.push(inner.trim());
    return "";
  });
  const pre: Precontext[] = [];
  text = text.replace(TAG_RE("precontext"), (_m, inner: string) => {
    try {
      const parsed = JSON.parse(inner.trim());
      if (Array.isArray(parsed)) pre.push(...parsed);
      else pre.push(parsed);
    } catch {
      /* ignore malformed block */
    }
    return "";
  });
  const out: { text: string; reasoning?: string; precontext?: Precontext[] } = { text: text.trim() };
  if (thinks.length) out.reasoning = thinks.join("\n");
  if (pre.length) out.precontext = pre;
  return out;
}

const SIDE_OPEN = ["<think>", "<precontext>"] as const;
const SIDE_CLOSE: Record<string, string> = { "<think>": "</think>", "<precontext>": "</precontext>" };

function suffixPrefixLen(s: string, tag: string): number {
  for (let k = Math.min(s.length, tag.length - 1); k > 0; k--) {
    if (s.slice(s.length - k) === tag.slice(0, k)) return k;
  }
  return 0;
}

/** Strips inline `<think>`/`<precontext>` blocks from streamed content, chunk by chunk. */
export class SideChannelFilter {
  #buf = "";
  #close: string | undefined;

  feed(text: string): string {
    this.#buf += text;
    const out: string[] = [];
    while (this.#buf) {
      if (this.#close === undefined) {
        const lt = this.#buf.indexOf("<");
        if (lt === -1) {
          out.push(this.#buf);
          this.#buf = "";
          break;
        }
        if (lt > 0) {
          out.push(this.#buf.slice(0, lt));
          this.#buf = this.#buf.slice(lt);
        }
        const opened = SIDE_OPEN.find((t) => this.#buf.startsWith(t));
        if (opened) {
          this.#close = SIDE_CLOSE[opened];
          this.#buf = this.#buf.slice(opened.length);
          continue;
        }
        if (SIDE_OPEN.some((t) => t.startsWith(this.#buf))) break;
        out.push("<");
        this.#buf = this.#buf.slice(1);
      } else {
        const close = this.#close;
        const end = this.#buf.indexOf(close);
        if (end === -1) {
          const keep = suffixPrefixLen(this.#buf, close);
          this.#buf = keep ? this.#buf.slice(this.#buf.length - keep) : "";
          break;
        }
        this.#buf = this.#buf.slice(end + close.length);
        this.#close = undefined;
      }
    }
    return out.join("");
  }

  flush(): string {
    if (this.#close !== undefined) {
      this.#buf = "";
      return "";
    }
    const rest = this.#buf;
    this.#buf = "";
    return rest;
  }
}
