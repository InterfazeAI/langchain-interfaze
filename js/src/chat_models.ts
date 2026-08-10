import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { type LangSmithParams } from "@langchain/core/language_models/chat_models";
import { convertChunksToEvents } from "@langchain/core/language_models/compat";
import type { ChatModelStreamEvent, FinishReason } from "@langchain/core/language_models/event";
import { AIMessage, AIMessageChunk, type BaseMessage, isAIMessage } from "@langchain/core/messages";
import { ChatGenerationChunk, type ChatResult } from "@langchain/core/outputs";
import { concat } from "@langchain/core/utils/stream";
import { ChatOpenAICompletions, type ChatOpenAIFields, normalizeHeaders } from "@langchain/openai";
import { INTERFAZE_BASE_URL, INTERFAZE_MODEL, InterfazeError } from "interfaze";
import { SideChannelFilter, stripSideChannels, TAG_RE } from "./side_channels.js";
import { VERSION } from "./version.js";

const PROVIDER = "interfaze";

// The v3 protocol has its own vocabulary; anything unmapped leaves `reason` untouched.
const FINISH_REASONS: Record<string, FinishReason> = {
  stop: "stop",
  length: "length",
  tool_calls: "tool_use",
  function_call: "tool_use",
  content_filter: "content_filter",
};

const DEFAULT_TIMEOUT_MS = 900_000;

const HEADER_SHOW_ADDITIONAL_INFO = "x-show-additional-info";
const HEADER_BYPASS_MOA = "x-interfaze-bypass-moa";
const HEADER_BYPASS_CACHE = "x-interfaze-bypass-cache";

export type InterfazeReasoningEffort = "minimal" | "low" | "medium" | "high" | "on" | "off" | "auto";

export interface ChatInterfazeFields extends Omit<ChatOpenAIFields, "reasoningEffort"> {
  apiKey?: string;
  reasoningEffort?: InterfazeReasoningEffort;
  /** Stream `<precontext>` deltas (`x-show-additional-info`); the only way to get
   *  precontext while streaming, since non-streaming responses always carry it. */
  showAdditionalInfo?: boolean;
  /** Skip the mixture-of-architecture internal tool router (`x-interfaze-bypass-moa`). */
  bypassMoA?: boolean;
  /** Skip the semantic cache (`x-interfaze-bypass-cache`). */
  bypassCache?: boolean;
}

type VideoBlock = {
  type: "video";
  url?: string;
  base64?: string;
  file_id?: string;
  mime_type?: string;
  extras?: { filename?: string };
};

const VIDEO_MIME: Record<string, string> = {
  mp4: "video/mp4",
  mov: "video/quicktime",
  webm: "video/webm",
  avi: "video/x-msvideo",
  mkv: "video/x-matroska",
  "3gp": "video/3gpp",
};

function videoMimeFromUrl(url: string): string | undefined {
  const base = url.split("?")[0]!.split("#")[0]!;
  const ext = base.includes(".") ? base.slice(base.lastIndexOf(".") + 1).toLowerCase() : "";
  return VIDEO_MIME[ext];
}

function convertVideoBlock(block: VideoBlock): Record<string, unknown> {
  if (block.file_id != null) {
    throw new InterfazeError("Interfaze cannot resolve a video by 'file_id'. Pass 'url' or 'base64' instead.");
  }
  let mime = block.mime_type;
  let file: Record<string, unknown>;
  if (block.url != null) {
    file = { file_data: block.url };
    mime = mime || videoMimeFromUrl(block.url);
  } else if (block.base64 != null) {
    mime = mime || "video/mp4";
    file = { file_data: `data:${mime};base64,${block.base64}` };
  } else {
    throw new InterfazeError("Video content block requires one of 'url' or 'base64'.");
  }
  if (mime) file.format = mime;
  const filename = block.extras?.filename;
  if (filename) file.filename = filename;
  return { type: "file", file };
}

const SIDE_FIELDS = ["precontext", "reasoning", "vcache"] as const;

type SideChannelCarrier = {
  content: unknown;
  response_metadata: Record<string, unknown>;
  additional_kwargs: Record<string, unknown>;
};

const carriesValue = (value: unknown): boolean => value !== undefined && value !== null && value !== "";

/** `[]` is truthy in JS, so a bare Boolean() would let an empty precontext block the real one. */
const hasValue = (value: unknown): boolean => (Array.isArray(value) ? value.length > 0 : Boolean(value));

const ACCUMULATING_SIDE_FIELDS: readonly string[] = ["precontext", "reasoning"];

// Accumulating fields dedupe by value, so a different payload still lands. `vcache` is
// scalar state and dedupes by name — merging two values would concatenate them.
const stableStringify = (value: unknown): string =>
  JSON.stringify(value, (_k, v) =>
    v && typeof v === "object" && !Array.isArray(v)
      ? Object.fromEntries(Object.entries(v as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b)))
      : v
  );

// Matches python's json.dumps(sort_keys=True): the same payload with reordered keys
// must dedupe, not double-emit.
const fingerprint = (key: string, value: unknown): string => (ACCUMULATING_SIDE_FIELDS.includes(key) ? `${key}:${stableStringify(value)}` : key);

function applySideFields(message: SideChannelCarrier, raw: Record<string, unknown>, seen?: Set<string>): void {
  for (const key of SIDE_FIELDS) {
    const value = raw[key];
    if (!carriesValue(value)) continue;
    const fp = fingerprint(key, value);
    if (seen?.has(fp)) continue;
    seen?.add(fp);
    message.response_metadata[key] = value;
    message.additional_kwargs[key] = value as never;
  }
}

/** Closed blocks removed, no trim — `stripSideChannels` trims and breaks prefix compares. */
function withoutClosedBlocks(raw: string): string {
  return raw.replace(TAG_RE("think"), "").replace(TAG_RE("precontext"), "");
}

/**
 * Earliest unmatched opening tag, by position rather than tag order — a truncated answer
 * whose prose mentions `<think>` before an unclosed `<precontext>` must split at the
 * precontext. Callers gate on truncation: in a completed response an unmatched tag is
 * prose the model wrote, not a channel the server failed to close.
 */
function openSideChannel(text: string): { tag: "think" | "precontext"; before: string; after: string } | null {
  const found = (["think", "precontext"] as const)
    .map((tag) => ({ tag, at: text.indexOf(`<${tag}>`) }))
    .filter(({ at }) => at !== -1)
    .sort((a, b) => a.at - b.at);
  const first = found[0];
  if (!first) return null;
  return { tag: first.tag, before: text.slice(0, first.at), after: text.slice(first.at + first.tag.length + 2) };
}

/**
 * What the caller still owes, given what already streamed. On a truncated response the
 * partial `<think>` becomes reasoning rather than content, and a partial `<precontext>`
 * — unparseable tool JSON — is dropped outright.
 */
function recoverTail(raw: string, emitted: string, truncated: boolean): { tail: string; reasoning?: string } {
  const text = withoutClosedBlocks(raw);
  const open = truncated ? openSideChannel(text) : null;
  const visible = open ? open.before : text;
  const tail = visible.startsWith(emitted) ? visible.slice(emitted.length) : "";
  return open?.tag === "think" && open.after ? { tail, reasoning: open.after } : { tail };
}

function stripTags(message: AIMessage, truncated = false): void {
  if (typeof message.content !== "string") return;
  if (!message.content.includes("<think>") && !message.content.includes("<precontext>")) return;
  const stripped = stripSideChannels(message.content);
  const recovered = truncated ? recoverTail(message.content, "", true) : null;
  const text = recovered ? recovered.tail.trim() : stripped.text;
  const reasoning = stripped.reasoning || recovered?.reasoning;
  const { precontext } = stripped;
  if (text !== message.content) message.content = text;
  if (reasoning && !hasValue(message.response_metadata.reasoning)) {
    message.response_metadata.reasoning = reasoning;
    message.additional_kwargs.reasoning = reasoning as never;
  }
  if (precontext && !hasValue(message.response_metadata.precontext)) {
    message.response_metadata.precontext = precontext;
    message.additional_kwargs.precontext = precontext as never;
  }
}

function rewriteContent(content: unknown): unknown {
  if (!Array.isArray(content)) return content;
  let changed = false;
  const out = content.map((block) => {
    if (!block || typeof block !== "object") return block;
    if ((block as { type?: string }).type === "video") {
      changed = true;
      return convertVideoBlock(block as VideoBlock);
    }
    // Interfaze has no file store, so a file_id reference can only 400 downstream.
    // Both the standard block shape and the openai-native nesting under `file`.
    const nested = (block as { file?: { file_id?: unknown } }).file?.file_id;
    if ((block as { file_id?: unknown }).file_id != null || nested != null) {
      throw new InterfazeError("Interfaze cannot resolve content by 'file_id'. Pass 'url' or 'base64' instead.");
    }
    return block;
  });
  return changed ? out : content;
}

const PUBLIC_HEADERS: readonly string[] = [HEADER_SHOW_ADDITIONAL_INFO, HEADER_BYPASS_MOA, HEADER_BYPASS_CACHE];

/** FNV-1a because no sync cryptographic hash exists in every runtime this package runs in.
 *  Not a security boundary: distinctness is all the cache key and the trace need. */
const digest = (value: string): string => {
  let hash = 0x811c9dc5;
  for (let i = 0; i < value.length; i += 1) hash = Math.imul(hash ^ value.charCodeAt(i), 0x01000193);
  return (hash >>> 0).toString(16);
};

const redactHeaders = (headers: Record<string, string>): string[] =>
  Object.keys(headers)
    .sort()
    .map((key) => (PUBLIC_HEADERS.includes(key) ? `${key}=${headers[key]}` : `${key}#${digest(String(headers[key]))}`));

function buildHeaders(fields: ChatInterfazeFields): Record<string, string> | undefined {
  // defaultHeaders is HeadersLike: spreading a Headers instance yields {} and a tuple
  // array yields {"0": [k, v]}. normalizeHeaders also lowercases, so a differently-cased
  // caller header is replaced rather than concatenated onto ours.
  const given = fields.configuration?.defaultHeaders;
  const headers = normalizeHeaders(given) as Record<string, string>;
  // normalizeHeaders keeps string values only, so `{"x-flag": true}` would vanish silently.
  if (given && typeof given === "object" && !Array.isArray(given) && !(given instanceof Headers)) {
    for (const [key, value] of Object.entries(given)) {
      if (typeof value === "number" || typeof value === "boolean") headers[key.toLowerCase()] = String(value);
    }
  }
  if (fields.showAdditionalInfo) headers[HEADER_SHOW_ADDITIONAL_INFO] = "true";
  if (fields.bypassMoA) headers[HEADER_BYPASS_MOA] = "true";
  if (fields.bypassCache) headers[HEADER_BYPASS_CACHE] = "true";
  return Object.keys(headers).length ? headers : undefined;
}

export class ChatInterfaze extends ChatOpenAICompletions {
  static override lc_name(): string {
    return "ChatInterfaze";
  }

  // Provider family, not the model: `interfaze-beta` reaches tracing and the cache key
  // via `ls_model_name` / `model_name`. Mirrors ChatOpenAI's "openai-chat".
  override _llmType(): string {
    return "interfaze";
  }

  override lc_namespace = ["langchain", "chat_models", PROVIDER];

  override get lc_secrets(): { [key: string]: string } {
    return { apiKey: "INTERFAZE_API_KEY" };
  }

  /** Kept off the parent, whose `reasoningEffort` type is narrower than Interfaze accepts. */
  readonly interfazeReasoningEffort?: InterfazeReasoningEffort;

  constructor(fields: ChatInterfazeFields = {}) {
    const { apiKey, model, configuration, timeout, showAdditionalInfo, bypassMoA, bypassCache, reasoningEffort, ...rest } = fields;
    const key = apiKey ?? (typeof process !== "undefined" ? process.env?.INTERFAZE_API_KEY : undefined);
    if (!key) {
      throw new InterfazeError("Missing API key. Pass new ChatInterfaze({ apiKey: ... }) or set the INTERFAZE_API_KEY environment variable.");
    }
    const defaultHeaders = buildHeaders(fields);
    super({
      ...rest,
      apiKey: key,
      model: model ?? INTERFAZE_MODEL,
      timeout: timeout ?? DEFAULT_TIMEOUT_MS,
      configuration: {
        baseURL: INTERFAZE_BASE_URL,
        ...configuration,
        ...(defaultHeaders ? { defaultHeaders } : {}),
      },
      __includeRawResponse: true,
    });
    this.lc_serializable = false;
    this.interfazeReasoningEffort = reasoningEffort;
    this._addVersion("@interfaze-ai/langchain", VERSION);
  }

  // The parent spreads clientConfig wholesale, landing the api key and every default
  // header verbatim in the llm cache key. Fingerprint them instead.
  override _identifyingParams(): ReturnType<ChatOpenAICompletions["_identifyingParams"]> {
    const { apiKey, defaultHeaders, ...rest } = super._identifyingParams();
    return {
      ...rest,
      ...(typeof apiKey === "string" ? { interfazeKey: digest(apiKey) } : {}),
      ...(defaultHeaders ? { interfazeHeaders: redactHeaders(defaultHeaders as Record<string, string>) } : {}),
    } as ReturnType<ChatOpenAICompletions["_identifyingParams"]>;
  }

  override getLsParams(options: this["ParsedCallOptions"]): LangSmithParams {
    return { ...super.getLsParams(options), ls_provider: PROVIDER };
  }

  override invocationParams(
    options?: this["ParsedCallOptions"],
    extra?: { streaming?: boolean }
  ): ReturnType<ChatOpenAICompletions["invocationParams"]> {
    const params = super.invocationParams(options, extra);
    const opts = options as { reasoningEffort?: InterfazeReasoningEffort; reasoning?: { effort?: InterfazeReasoningEffort } } | undefined;
    const effort =
      opts?.reasoning?.effort ??
      opts?.reasoningEffort ??
      (this.reasoning?.effort as InterfazeReasoningEffort | null | undefined) ??
      this.interfazeReasoningEffort;
    if (effort != null) params.reasoning_effort = effort as NonNullable<typeof params.reasoning_effort>;
    return params;
  }

  // Interfaze sends `role` on the first delta only. If a stream ever opens without one the
  // parent picks ChatMessageChunk, which fails isAIMessage() and carries no
  // additional_kwargs, so __raw_response and every side field are lost.
  protected override _convertCompletionsDeltaToBaseMessageChunk(
    delta: Record<string, any>,
    rawResponse: any,
    defaultRole?: any
  ): ReturnType<ChatOpenAICompletions["_convertCompletionsDeltaToBaseMessageChunk"]> {
    return super._convertCompletionsDeltaToBaseMessageChunk(delta, rawResponse, defaultRole ?? "assistant");
  }

  // The parent drops choice-less frames, hiding side fields that ride a usage-only frame.
  // Observed rather than reshaped: injecting a choice would duplicate the usage envelope.
  override async completionWithRetry(request: any, requestOptions?: any): Promise<any> {
    const result = await super.completionWithRetry(request, requestOptions);
    const sink = requestOptions && this.#frameSinks.get(requestOptions);
    if (!request?.stream || !sink) return result;
    const frames = result as AsyncIterable<Record<string, unknown>>;
    return (async function* () {
      for await (const frame of frames) {
        if (SIDE_FIELDS.some((k) => carriesValue(frame[k]))) sink.push(frame);
        yield frame;
      }
    })();
  }

  // Keyed on the call options, the one object the parent hands back to
  // completionWithRetry, so concurrent streams never share a sink.
  readonly #frameSinks = new WeakMap<object, Array<Record<string, unknown>>>();

  private rewriteVideoBlocks(messages: BaseMessage[]): BaseMessage[] {
    return messages.map((m) => {
      if (!Array.isArray(m.content)) return m;
      const rewritten = rewriteContent(m.content);
      if (rewritten === m.content) return m;
      const copy = Object.create(Object.getPrototypeOf(m));
      Object.assign(copy, m);
      copy.content = rewritten;
      return copy as BaseMessage;
    });
  }

  override async _generate(messages: BaseMessage[], options: this["ParsedCallOptions"], runManager?: CallbackManagerForLLMRun): Promise<ChatResult> {
    const result = await super._generate(this.rewriteVideoBlocks(messages), options, runManager);
    for (const generation of result.generations) {
      const message = generation.message;
      if (isAIMessage(message)) {
        message.response_metadata.model_provider = PROVIDER;
        const raw = message.additional_kwargs.__raw_response as Record<string, unknown> | undefined;
        if (raw) applySideFields(message, raw);
        delete message.additional_kwargs.__raw_response;
        stripTags(message, generation.generationInfo?.finish_reason === "length");
        if (typeof message.content === "string") generation.text = message.content;
      }
    }
    return result;
  }

  override async *_streamResponseChunks(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatGenerationChunk> {
    const filter = new SideChannelFilter();
    const rawParts: string[] = [];
    const emittedParts: string[] = [];
    const seen = new Set<string>();
    const frames: Array<Record<string, unknown>> = [];
    this.#frameSinks.set(options, frames);
    let streamId: string | undefined;
    let finishReason: unknown;
    const sideChunk = (side: Record<string, unknown>): ChatGenerationChunk | null => {
      const message = new AIMessageChunk({ content: "", id: streamId });
      applySideFields(message, side, seen);
      if (Object.keys(message.additional_kwargs).length === 0) return null;
      message.response_metadata.model_provider = PROVIDER;
      return new ChatGenerationChunk({ message, text: "" });
    };
    for await (const gen of super._streamResponseChunks(this.rewriteVideoBlocks(messages), options, runManager)) {
      const message = gen.message as unknown as SideChannelCarrier;
      streamId ??= (gen.message as AIMessageChunk).id;
      finishReason = gen.generationInfo?.finish_reason ?? finishReason;
      message.response_metadata.model_provider = PROVIDER;
      const raw = message.additional_kwargs.__raw_response as Record<string, unknown> | undefined;
      if (raw) applySideFields(message, raw, seen);
      delete message.additional_kwargs.__raw_response;
      if (typeof message.content === "string" && message.content) {
        rawParts.push(message.content);
        const filtered = filter.feed(message.content);
        emittedParts.push(filtered);
        message.content = filtered;
        // handleLLMNewToken fires after the yield and reads gen.text, not
        // message.content, so keep it in sync or callbacks see the raw tags.
        gen.text = filtered;
      }
      // Emitted inline rather than at stream end so a consumer that breaks early still
      // sees what the server had already sent, matching python's chunk positions.
      for (const frame of frames.splice(0)) {
        const side = sideChunk(frame);
        if (side) yield side;
      }
      yield gen;
    }
    const joined = rawParts.join("");
    this.#frameSinks.delete(options);
    const flushed = filter.flush();
    const recovered = flushed ? { tail: flushed } : recoverTail(joined, emittedParts.join(""), finishReason === "length");
    const tail = recovered.tail;
    const stripped = stripSideChannels(joined);
    const { precontext } = stripped;
    const reasoning = stripped.reasoning || recovered.reasoning;
    if (tail) {
      const message = new AIMessageChunk({ content: tail, id: streamId });
      message.response_metadata.model_provider = PROVIDER;
      const chunk = new ChatGenerationChunk({ message, text: tail });
      yield chunk;
      await runManager?.handleLLMNewToken(tail, { prompt: 0, completion: 0 }, undefined, undefined, undefined, { chunk });
    }
    const inline: Record<string, unknown> = {};
    if (reasoning) inline.reasoning = reasoning;
    if (precontext) inline.precontext = precontext;
    // One chunk per source, so langchain's own merge concatenates them — the same
    // behaviour the python package gets for free from its per-chunk conversion.
    for (const side of [...frames.splice(0), inline]) {
      const chunk = sideChunk(side);
      if (chunk) yield chunk;
    }
  }

  /**
   * Neither inherited implementation works: the parent reads the raw stream so `<think>`
   * leaks into events, while `convertChunksToEvents` strips tags but hardcodes
   * `reason: "stop"` and emits no `responseMetadata`. Convert our own filtered chunks and
   * restore the metadata on the terminal event.
   */
  override async *_streamChatModelEvents(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatModelStreamEvent> {
    const responseMetadata: Record<string, unknown> = {};
    let merged: AIMessageChunk | undefined;
    const source = this._streamResponseChunks(messages, options, runManager);
    const observed = (async function* () {
      for await (const gen of source) {
        const message = gen.message as AIMessageChunk;
        // Last wins for everything the server restates per frame. Merging those instead
        // would sum them: the parent puts `usage` on two chunks, and langchain's merge
        // adds numbers, so the terminal event would report double the tokens.
        Object.assign(responseMetadata, message.response_metadata);
        merged = merged ? concat(merged, message) : message;
        for (const key of ["finish_reason", "model_name"] as const) {
          const value = gen.generationInfo?.[key];
          if (value != null) responseMetadata[key] = value;
        }
        yield gen;
      }
    })();
    for await (const event of convertChunksToEvents(observed, { signal: options.signal })) {
      if (event.event !== "message-finish") {
        yield event;
        continue;
      }
      // ...except the accumulating fields, whose one-chunk-per-source emission exists so
      // langchain's merge concatenates them, which is what `.stream()` consumers see.
      for (const key of ACCUMULATING_SIDE_FIELDS) {
        const value = merged?.response_metadata[key];
        if (value != null) responseMetadata[key] = value;
      }
      const reason = FINISH_REASONS[String(responseMetadata.finish_reason)];
      yield { ...event, ...(reason ? { reason } : {}), responseMetadata };
    }
  }
}
