import { ChatOpenAICompletions, type ChatOpenAIFields } from "@langchain/openai";
import { INTERFAZE_BASE_URL, INTERFAZE_MODEL, InterfazeError } from "interfaze";

export interface ChatInterfazeFields extends ChatOpenAIFields {
  /** Interfaze API key; falls back to `process.env.INTERFAZE_API_KEY`. */
  apiKey?: string;
  /** Precomputed tool output passed to Interfaze to skip its internal tool run. */
  precontext?: Array<Record<string, unknown>>;
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
}
