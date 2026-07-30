import { ChatOpenAICompletions, type ChatOpenAIFields } from "@langchain/openai";
import { INTERFAZE_BASE_URL, INTERFAZE_MODEL, InterfazeError } from "interfaze";
import { AIMessage, AIMessageChunk, type BaseMessage } from "@langchain/core/messages";
import { ChatGenerationChunk, type ChatResult } from "@langchain/core/outputs";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";
import { SideChannelFilter, stripSideChannels } from "./side_channels.js";

export interface ChatInterfazeFields extends ChatOpenAIFields {
  /** Interfaze API key; falls back to `process.env.INTERFAZE_API_KEY`. */
  apiKey?: string;
  /** Precomputed tool output passed to Interfaze to skip its internal tool run. */
  precontext?: Array<Record<string, unknown>>;
}

type VideoBlock = { type: "video"; url?: string; base64?: string; file_id?: string; mime_type?: string; extras?: { filename?: string } };

function convertVideoBlock(block: VideoBlock): Record<string, unknown> {
  let mime = block.mime_type;
  let file: Record<string, unknown>;
  // Key-existence checks (not truthiness) to match Python's `"url" in block`.
  if ("url" in block) {
    file = { file_data: block.url };
  } else if ("base64" in block) {
    mime = mime ?? "video/mp4";
    file = { file_data: `data:${mime};base64,${block.base64}` };
  } else if ("file_id" in block) {
    file = { file_id: block.file_id };
  } else {
    throw new InterfazeError("Video content block requires one of 'url', 'base64', or 'file_id'.");
  }
  // Python stamps `format` whenever mime is truthy (always, for base64). Match it.
  if (mime) file.format = mime;
  const filename = block.extras?.filename;
  if (filename) file.filename = filename;
  return { type: "file", file };
}

const SIDE_FIELDS = ["precontext", "reasoning", "vcache"] as const;

function applySideFields(message: AIMessage, raw: Record<string, unknown>): void {
  for (const key of SIDE_FIELDS) {
    const value = raw[key];
    if (value !== undefined && value !== null) {
      message.response_metadata[key] = value;
      message.additional_kwargs[key] = value as never;
    }
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

export class ChatInterfaze extends ChatOpenAICompletions {
  constructor(fields: ChatInterfazeFields = {}) {
    const { apiKey, precontext, model, configuration, modelKwargs, ...rest } = fields;
    const key = apiKey ?? process.env.INTERFAZE_API_KEY;
    if (!key) {
      throw new InterfazeError("Missing API key. Pass new ChatInterfaze({ apiKey: ... }) or set the INTERFAZE_API_KEY environment variable.");
    }
    super({
      ...rest,
      apiKey: key,
      model: model ?? INTERFAZE_MODEL,
      configuration: { baseURL: INTERFAZE_BASE_URL, ...configuration },
      modelKwargs: precontext !== undefined ? { ...modelKwargs, precontext } : modelKwargs,
      __includeRawResponse: true,
    });
    // BaseChatOpenAI sets lc_serializable = true; override for Python SDK parity
    // (this class is not intended to round-trip through LangChain's serialization).
    this.lc_serializable = false;
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
    for await (const gen of super._streamResponseChunks(this.rewriteVideoBlocks(messages), options, runManager)) {
      const message = gen.message;
      if (message instanceof AIMessageChunk) {
        // Python parity: read top-level side-fields off each chunk's raw, then
        // strip __raw_response so it never leaks (see Task 6's no-leak rule).
        const raw = message.additional_kwargs.__raw_response as Record<string, unknown> | undefined;
        if (raw) applySideFields(message, raw);
        delete message.additional_kwargs.__raw_response;
        if (typeof message.content === "string" && message.content) {
          rawParts.push(message.content);
          message.content = filter.feed(message.content);
          // `gen.text` mirrors `message.content` in the upstream OpenAI integration and is
          // read independently by callback consumers (e.g. handleLLMNewToken's token arg,
          // legacy streamEvents v1 on_llm_end). Keep it filtered too so raw tags can't leak
          // through that side door.
          gen.text = message.content;
        }
      }
      yield gen;
    }
    const tail = filter.flush();
    const { reasoning, precontext } = stripSideChannels(rawParts.join(""));
    if (!tail && !reasoning && !precontext) return;
    const finalMessage = new AIMessageChunk({ content: tail });
    if (reasoning) {
      finalMessage.response_metadata.reasoning = reasoning;
      finalMessage.additional_kwargs.reasoning = reasoning;
    }
    if (precontext) {
      finalMessage.response_metadata.precontext = precontext;
      finalMessage.additional_kwargs.precontext = precontext as never;
    }
    yield new ChatGenerationChunk({ message: finalMessage, text: tail });
  }
}
