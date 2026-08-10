export type Precontext = Record<string, unknown>;

export const TAG_RE = (tag: string) => new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`, "g");

/** Pull `<think>`/`<precontext>` blocks out of content; returns the rest as `text`. */
export function stripSideChannels(content: string): {
  text: string;
  reasoning?: string;
  precontext?: Precontext[];
} {
  let text = content;
  const thinkBlocks: string[] = [];
  text = text.replace(TAG_RE("think"), (_m, inner: string) => {
    thinkBlocks.push(inner.trim());
    return "";
  });
  const precontexts: Precontext[] = [];
  text = text.replace(TAG_RE("precontext"), (_m, inner: string) => {
    try {
      const parsed = JSON.parse(inner.trim());
      if (Array.isArray(parsed)) precontexts.push(...parsed);
      else precontexts.push(parsed);
    } catch {
      /* ignore malformed block */
    }
    return "";
  });
  const out: { text: string; reasoning?: string; precontext?: Precontext[] } = { text: text.trim() };
  if (thinkBlocks.length) out.reasoning = thinkBlocks.join("\n");
  if (precontexts.length) out.precontext = precontexts;
  return out;
}

const SIDE_OPEN = ["<think>", "<precontext>"] as const;
const SIDE_CLOSE: Record<string, string> = { "<think>": "</think>", "<precontext>": "</precontext>" };

/** Length of the longest suffix of `text` that opens `tag`, so a tag split across two
 *  chunks is held back rather than emitted as content. */
function danglingTagPrefixLength(text: string, tag: string): number {
  for (let k = Math.min(text.length, tag.length - 1); k > 0; k--) {
    if (text.slice(text.length - k) === tag.slice(0, k)) return k;
  }
  return 0;
}

/** Strips inline `<think>`/`<precontext>` blocks from streamed content, chunk by chunk. */
export class SideChannelFilter {
  #buffer = "";
  #close: string | undefined;

  feed(text: string): string {
    this.#buffer += text;
    const out: string[] = [];
    while (this.#buffer) {
      if (this.#close === undefined) {
        const openAngleAt = this.#buffer.indexOf("<");
        if (openAngleAt === -1) {
          out.push(this.#buffer);
          this.#buffer = "";
          break;
        }
        if (openAngleAt > 0) {
          out.push(this.#buffer.slice(0, openAngleAt));
          this.#buffer = this.#buffer.slice(openAngleAt);
        }
        const opened = SIDE_OPEN.find((t) => this.#buffer.startsWith(t));
        if (opened) {
          this.#close = SIDE_CLOSE[opened];
          this.#buffer = this.#buffer.slice(opened.length);
          continue;
        }
        if (SIDE_OPEN.some((t) => t.startsWith(this.#buffer))) break;
        out.push("<");
        this.#buffer = this.#buffer.slice(1);
      } else {
        const close = this.#close;
        const end = this.#buffer.indexOf(close);
        if (end === -1) {
          const keep = danglingTagPrefixLength(this.#buffer, close);
          this.#buffer = keep ? this.#buffer.slice(this.#buffer.length - keep) : "";
          break;
        }
        this.#buffer = this.#buffer.slice(end + close.length);
        this.#close = undefined;
      }
    }
    return out.join("");
  }

  flush(): string {
    if (this.#close !== undefined) {
      this.#buffer = "";
      return "";
    }
    const rest = this.#buffer;
    this.#buffer = "";
    return rest;
  }
}
