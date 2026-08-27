/**
 * The official [LangChain](https://js.langchain.com) integration for
 * [Interfaze](https://interfaze.ai).
 *
 * {@link ChatInterfaze} is a standard LangChain chat model that talks to the
 * Interfaze endpoint. It forwards the usual chat-model options and additionally
 * surfaces the side channels Interfaze returns — `precontext`, `reasoning`, and
 * `vcache` — on both `response_metadata` and `additional_kwargs`.
 *
 * ```ts
 * import { ChatInterfaze } from "@interfaze/langchain";
 *
 * // Reads INTERFAZE_API_KEY from the environment, or pass { apiKey }.
 * const llm = new ChatInterfaze();
 * const res = await llm.invoke("Which US public companies reported earnings today?");
 * console.log(res.content);
 * console.log(res.response_metadata.precontext);
 * ```
 *
 * @module
 */

export { ChatInterfaze } from "./chat_models.js";
export type { ChatInterfazeFields, InterfazeReasoningEffort } from "./chat_models.js";
