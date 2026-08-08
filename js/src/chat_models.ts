import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { BaseChatModel, type LangSmithParams } from "@langchain/core/language_models/chat_models";
import type { ChatModelStreamEvent } from "@langchain/core/language_models/event";
import { AIMessage, AIMessageChunk, type BaseMessage } from "@langchain/core/messages";
import { ChatGenerationChunk, type ChatResult } from "@langchain/core/outputs";
import { ChatOpenAICompletions, type ChatOpenAIFields } from "@langchain/openai";
import { INTERFAZE_BASE_URL, INTERFAZE_MODEL, InterfazeError } from "interfaze";
import { SideChannelFilter, stripSideChannels } from "./side_channels.js";
import { VERSION } from "./version.js";

const PROVIDER = "interfaze";

const DEFAULT_TIMEOUT_MS = 900_000;

const HEADER_SHOW_ADDITIONAL_INFO = "x-show-additional-info";
const HEADER_BYPASS_MOA = "x-interfaze-bypass-moa";
const HEADER_BYPASS_CACHE = "x-interfaze-bypass-cache";

export type InterfazeReasoningEffort = "minimal" | "low" | "medium" | "high" | "on" | "off" | "auto";

export interface ChatInterfazeFields extends Omit<ChatOpenAIFields, "reasoningEffort"> {
  apiKey?: string;
  reasoningEffort?: InterfazeReasoningEffort;
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
  if (block.file_id !== undefined) {
    throw new InterfazeError("Interfaze cannot resolve a video by 'file_id'. Pass 'url' or 'base64' instead.");
  }
  let mime = block.mime_type;
  let file: Record<string, unknown>;
  if (block.url !== undefined) {
    file = { file_data: block.url };
    mime = mime ?? videoMimeFromUrl(block.url);
  } else if (block.base64 !== undefined) {
    mime = mime ?? "video/mp4";
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

const carriesValue = (value: unknown): boolean =>
  value !== undefined && value !== null && value !== "" && !(Array.isArray(value) && value.length === 0);

const ACCUMULATING_SIDE_FIELDS: readonly string[] = ["precontext", "reasoning"];

// Accumulating fields dedupe by value, so a different payload still lands. `vcache` is
// scalar state and dedupes by name — merging two values would concatenate them.
const fingerprint = (key: string, value: unknown): string => (ACCUMULATING_SIDE_FIELDS.includes(key) ? `${key}:${JSON.stringify(value)}` : key);

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

/**
 * A truncated response leaves a tag open; the filter buffers everything after it and
 * drops it on flush. `after` is exactly that swallowed remainder — streaming has already
 * emitted `before`, so re-emitting it would duplicate the prefix.
 */
function unterminatedTag(raw: string): { before: string; after: string } | null {
  const text = stripSideChannels(raw).text;
  for (const tag of ["<think>", "<precontext>"]) {
    const start = text.indexOf(tag);
    if (start === -1 || text.slice(start).includes(`</${tag.slice(1)}`)) continue;
    // A half-written <precontext> is partial metadata JSON, not answer text — drop it.
    const after = tag === "<think>" ? text.slice(start + tag.length) : "";
    return { before: text.slice(0, start), after };
  }
  return null;
}

function stripTags(message: AIMessage): void {
  if (typeof message.content !== "string") return;
  if (!message.content.includes("<think>") && !message.content.includes("<precontext>")) return;
  const { text, reasoning, precontext } = stripSideChannels(message.content);
  const open = unterminatedTag(message.content);
  const visible = open ? (open.before + open.after).trim() : text;
  if (visible !== message.content) message.content = visible;
  if (reasoning && message.response_metadata.reasoning === undefined) {
    message.response_metadata.reasoning = reasoning;
    message.additional_kwargs.reasoning = reasoning as never;
  }
  if (precontext && message.response_metadata.precontext === undefined) {
    message.response_metadata.precontext = precontext;
    message.additional_kwargs.precontext = precontext as never;
  }
}

function rewriteContent(content: unknown): unknown {
  if (!Array.isArray(content)) return content;
  let changed = false;
  const out = content.map((block) => {
    if (block && typeof block === "object" && (block as { type?: string }).type === "video") {
      changed = true;
      return convertVideoBlock(block as VideoBlock);
    }
    return block;
  });
  return changed ? out : content;
}

function buildHeaders(fields: ChatInterfazeFields): Record<string, string> | undefined {
  const headers: Record<string, string> = { ...(fields.configuration?.defaultHeaders as Record<string, string>) };
  if (fields.showAdditionalInfo) headers[HEADER_SHOW_ADDITIONAL_INFO] = "true";
  if (fields.bypassMoA) headers[HEADER_BYPASS_MOA] = "true";
  if (fields.bypassCache) headers[HEADER_BYPASS_CACHE] = "true";
  return Object.keys(headers).length ? headers : undefined;
}

export class ChatInterfaze extends ChatOpenAICompletions {
  static override lc_name(): string {
    return "ChatInterfaze";
  }

  // A provider-family id, not a model id — `interfaze-beta` reaches tracing and the LLM
  // cache key via `ls_model_name` / `model_name`. Mirrors ChatOpenAI's "openai-chat".
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
    const key = apiKey ?? process.env.INTERFAZE_API_KEY;
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
    this._addVersion("@interfaze/langchain", VERSION);
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

  // Interfaze omits `role` on continuation deltas, and can omit it entirely. The parent
  // then picks ChatMessageChunk, which carries no additional_kwargs (so no
  // __raw_response) and fails isAIMessage(). Normalize at the source, as the core
  // interfaze SDKs do, rather than compensating downstream.
  protected override _convertCompletionsDeltaToBaseMessageChunk(
    delta: Record<string, any>,
    rawResponse: any,
    defaultRole?: any
  ): ReturnType<ChatOpenAICompletions["_convertCompletionsDeltaToBaseMessageChunk"]> {
    return super._convertCompletionsDeltaToBaseMessageChunk(delta, rawResponse, defaultRole ?? "assistant");
  }

  // The parent drops choice-less frames before building a chunk, so side fields riding a
  // usage-only frame are invisible downstream. Observe the raw frames rather than
  // reshaping them — injecting a choice would also duplicate the usage envelope.
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
      if (message instanceof AIMessage) {
        message.response_metadata.model_provider = PROVIDER;
        const raw = message.additional_kwargs.__raw_response as Record<string, unknown> | undefined;
        if (raw) applySideFields(message, raw);
        delete message.additional_kwargs.__raw_response;
        stripTags(message);
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
    const seen = new Set<string>();
    const frames: Array<Record<string, unknown>> = [];
    this.#frameSinks.set(options, frames);
    for await (const gen of super._streamResponseChunks(this.rewriteVideoBlocks(messages), options, runManager)) {
      const message = gen.message as unknown as SideChannelCarrier;
      message.response_metadata.model_provider = PROVIDER;
      const raw = message.additional_kwargs.__raw_response as Record<string, unknown> | undefined;
      if (raw) applySideFields(message, raw, seen);
      delete message.additional_kwargs.__raw_response;
      if (typeof message.content === "string" && message.content) {
        rawParts.push(message.content);
        const filtered = filter.feed(message.content);
        message.content = filtered;
        // handleLLMNewToken fires after the yield and reads gen.text, not
        // message.content, so keep it in sync or callbacks see the raw tags.
        gen.text = filtered;
      }
      yield gen;
    }
    const joined = rawParts.join("");
    const tail = filter.flush() || unterminatedTag(joined)?.after.trim() || "";
    const { reasoning, precontext } = stripSideChannels(joined);
    const emitReasoning = reasoning && !seen.has(fingerprint("reasoning", reasoning));
    const emitPrecontext = precontext && !seen.has(fingerprint("precontext", precontext));
    const leftover = new AIMessageChunk({ content: "" });
    for (const frame of frames) applySideFields(leftover, frame, seen);
    const hasLeftover = Object.keys(leftover.additional_kwargs).length > 0;
    if (!tail && !emitReasoning && !emitPrecontext && !hasLeftover) return;
    const finalMessage = new AIMessageChunk({ content: tail });
    finalMessage.response_metadata.model_provider = PROVIDER;
    Object.assign(finalMessage.response_metadata, leftover.response_metadata);
    Object.assign(finalMessage.additional_kwargs, leftover.additional_kwargs);
    if (emitReasoning) {
      finalMessage.response_metadata.reasoning = reasoning;
      finalMessage.additional_kwargs.reasoning = reasoning;
    }
    if (emitPrecontext) {
      finalMessage.response_metadata.precontext = precontext;
      finalMessage.additional_kwargs.precontext = precontext as never;
    }
    const finalChunk = new ChatGenerationChunk({ message: finalMessage, text: tail });
    yield finalChunk;
    await runManager?.handleLLMNewToken(tail, { prompt: 0, completion: 0 }, undefined, undefined, undefined, {
      chunk: finalChunk,
    });
  }

  override async *_streamChatModelEvents(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatModelStreamEvent> {
    yield* BaseChatModel.prototype._streamChatModelEvents.call(this, messages, options, runManager);
  }
}
