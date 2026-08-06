import { ChatOpenAICompletions, type ChatOpenAIFields } from "@langchain/openai";
import { INTERFAZE_BASE_URL, INTERFAZE_MODEL, InterfazeError } from "interfaze";
import { AIMessage, AIMessageChunk, type BaseMessage } from "@langchain/core/messages";
import { BaseChatModel, type LangSmithParams } from "@langchain/core/language_models/chat_models";
import { ChatGenerationChunk, type ChatResult } from "@langchain/core/outputs";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import type { ChatModelStreamEvent } from "@langchain/core/language_models/event";
import { SideChannelFilter, stripSideChannels } from "./side_channels.js";
import { VERSION } from "./version.js";

const PROVIDER = "interfaze";

/**
 * Interfaze runs OCR / web search / scraping / STT / forecasting inline, so a single
 * completion can legitimately take minutes. Matches the core `interfaze` SDK default.
 */
const DEFAULT_TIMEOUT_MS = 900_000;

/** Interfaze control-plane headers (mirrors the core `interfaze` SDK). */
const HEADER_SHOW_ADDITIONAL_INFO = "x-show-additional-info";
const HEADER_BYPASS_MOA = "x-interfaze-bypass-moa";
const HEADER_BYPASS_CACHE = "x-interfaze-bypass-cache";
const HEADER_ADMIN_KEY = "x-admin-key";

/** Wider than the OpenAI enum — Interfaze also accepts `on` / `off` / `auto`. */
export type InterfazeReasoningEffort = "minimal" | "low" | "medium" | "high" | "on" | "off" | "auto";

export interface ChatInterfazeFields extends Omit<ChatOpenAIFields, "reasoningEffort"> {
  /** Interfaze API key; falls back to `process.env.INTERFAZE_API_KEY`. */
  apiKey?: string;
  /**
   * Default reasoning effort for every call. `@langchain/openai` drops `reasoningEffort`
   * for model names it doesn't recognize as reasoning models, so this is forwarded here.
   */
  reasoningEffort?: InterfazeReasoningEffort;
  /**
   * Emit inline `<precontext>` blocks while streaming. Interfaze only sends streamed
   * precontext when this is on (`x-show-additional-info`).
   */
  showAdditionalInfo?: boolean;
  /** Skip the mixture-of-architecture internal tool router (`x-interfaze-bypass-moa`). */
  bypassMoA?: boolean;
  /** Skip the semantic cache (`x-interfaze-bypass-cache`). */
  bypassCache?: boolean;
  /** Admin key that surfaces a `debug` field (`x-admin-key`). */
  adminKey?: string;
}

type VideoBlock = {
  type: "video";
  url?: string;
  base64?: string;
  file_id?: string;
  mime_type?: string;
  extras?: { filename?: string };
};

/** Video containers Interfaze accepts, mirroring `interfaze`'s `inputs` helpers. */
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
  // Interfaze has no file store: the `file` part accepts `file_data` only.
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

function applySideFields(message: AIMessage, raw: Record<string, unknown>, seen?: Set<string>): void {
  for (const key of SIDE_FIELDS) {
    const value = raw[key];
    if (value === undefined || value === null) continue;
    // Chunks concatenate on aggregation, so a field repeated across chunks would be
    // duplicated (arrays) or string-concatenated (scalars). Emit each one once.
    if (seen?.has(key)) continue;
    seen?.add(key);
    message.response_metadata[key] = value;
    message.additional_kwargs[key] = value as never;
  }
}

function stripTags(message: AIMessage): void {
  if (typeof message.content !== "string") return;
  if (!message.content.includes("<think>") && !message.content.includes("<precontext>")) return;
  const { text, reasoning, precontext } = stripSideChannels(message.content);
  if (text !== message.content) message.content = text;
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
  if (fields.adminKey) headers[HEADER_ADMIN_KEY] = fields.adminKey;
  return Object.keys(headers).length ? headers : undefined;
}

export class ChatInterfaze extends ChatOpenAICompletions {
  static override lc_name(): string {
    return "ChatInterfaze";
  }

  override _llmType(): string {
    return "interfaze-chat";
  }

  override lc_namespace = ["langchain", "chat_models", PROVIDER];

  override get lc_secrets(): { [key: string]: string } {
    return { apiKey: "INTERFAZE_API_KEY" };
  }

  protected override get streamEventProvider(): string {
    return PROVIDER;
  }

  /** Kept out of the parent, whose type is narrower than what Interfaze accepts. */
  readonly interfazeReasoningEffort?: InterfazeReasoningEffort;

  constructor(fields: ChatInterfazeFields = {}) {
    const { apiKey, model, configuration, timeout, showAdditionalInfo, bypassMoA, bypassCache, adminKey, reasoningEffort, ...rest } = fields;
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

  /**
   * `@langchain/openai` only forwards `reasoningEffort` for model names matching its
   * own reasoning-model heuristic (`/^o\d/`, `gpt-5*`), so `interfaze-beta` would
   * silently lose it. Re-attach it here from the call options or the constructor.
   */
  override invocationParams(
    options?: this["ParsedCallOptions"],
    extra?: { streaming?: boolean }
  ): ReturnType<ChatOpenAICompletions["invocationParams"]> {
    const params = super.invocationParams(options, extra);
    const fromOptions = options as { reasoningEffort?: InterfazeReasoningEffort; reasoning?: { effort?: InterfazeReasoningEffort } } | undefined;
    const effort = fromOptions?.reasoningEffort ?? fromOptions?.reasoning?.effort ?? this.interfazeReasoningEffort;
    if (effort != null) params.reasoning_effort = effort as NonNullable<typeof params.reasoning_effort>;
    return params;
  }

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
    for await (const gen of super._streamResponseChunks(this.rewriteVideoBlocks(messages), options, runManager)) {
      const message = gen.message;
      if (message instanceof AIMessageChunk) {
        message.response_metadata.model_provider = PROVIDER;
        const raw = message.additional_kwargs.__raw_response as Record<string, unknown> | undefined;
        if (raw) applySideFields(message, raw, seen);
        delete message.additional_kwargs.__raw_response;
        if (typeof message.content === "string" && message.content) {
          rawParts.push(message.content);
          message.content = filter.feed(message.content);
          // handleLLMNewToken fires after the yield and reads gen.text, not
          // message.content, so keep it in sync or callbacks see the raw tags.
          gen.text = message.content;
        }
      }
      yield gen;
    }
    const tail = filter.flush();
    const { reasoning, precontext } = stripSideChannels(rawParts.join(""));
    if (!tail && !reasoning && !precontext) return;
    const finalMessage = new AIMessageChunk({ content: tail });
    finalMessage.response_metadata.model_provider = PROVIDER;
    if (reasoning && !seen.has("reasoning")) {
      finalMessage.response_metadata.reasoning = reasoning;
      finalMessage.additional_kwargs.reasoning = reasoning;
    }
    if (precontext && !seen.has("precontext")) {
      finalMessage.response_metadata.precontext = precontext;
      finalMessage.additional_kwargs.precontext = precontext as never;
    }
    const finalChunk = new ChatGenerationChunk({ message: finalMessage, text: tail });
    yield finalChunk;
    // super() fires this for every chunk it yields; the flushed tail is ours, so it
    // would otherwise never reach token-level callbacks.
    await runManager?.handleLLMNewToken(tail, { prompt: 0, completion: 0 }, undefined, undefined, undefined, {
      chunk: finalChunk,
    });
  }

  /**
   * `ChatOpenAICompletions` ships a native protocol-stream implementation that talks to
   * the wire directly and never calls `_streamResponseChunks`, so the side-channel
   * filter above would be skipped. Fall back to the generic `BaseChatModel` bridge,
   * which builds events from our filtered chunks.
   */
  override async *_streamChatModelEvents(
    messages: BaseMessage[],
    options: this["ParsedCallOptions"],
    runManager?: CallbackManagerForLLMRun
  ): AsyncGenerator<ChatModelStreamEvent> {
    yield* BaseChatModel.prototype._streamChatModelEvents.call(this, messages, options, runManager);
  }
}
