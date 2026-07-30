import { ChatOpenAICompletions, type ChatOpenAIFields } from "@langchain/openai";
import { INTERFAZE_BASE_URL, INTERFAZE_MODEL, InterfazeError } from "interfaze";
import type { BaseMessage } from "@langchain/core/messages";
import type { ChatResult } from "@langchain/core/outputs";
import type { CallbackManagerForLLMRun } from "@langchain/core/callbacks/manager";

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
  if (block.url) {
    file = { file_data: block.url };
  } else if (block.base64) {
    mime = mime ?? "video/mp4";
    file = { file_data: `data:${mime};base64,${block.base64}` };
  } else if (block.file_id) {
    file = { file_id: block.file_id };
  } else {
    throw new InterfazeError("Video content block requires one of 'url', 'base64', or 'file_id'.");
  }
  const filename = block.extras?.filename;
  if (filename) file.filename = filename;
  return { type: "file", file };
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
    return super._generate(this.rewriteVideoBlocks(messages), options, runManager);
  }
}
