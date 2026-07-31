import { ChatInterfaze, type ChatInterfazeFields } from "../src/index.js";

export const CHAT_URL = "https://api.interfaze.ai/v1/chat/completions";
export const VIDEO_URL = "https://download.samplelib.com/mp4/sample-5s.mp4";

export interface CapturedRequest {
  url: string;
  body: Record<string, unknown> | undefined;
}

/** Build a ChatInterfaze whose underlying client uses a capturing mock `fetch`. */
export function mockChat(
  responder: (req: CapturedRequest) => Response,
  extraFields: Partial<ChatInterfazeFields> = {}
): { model: ChatInterfaze; calls: CapturedRequest[] } {
  const calls: CapturedRequest[] = [];
  const fetchImpl = async (input: unknown, init: RequestInit = {}): Promise<Response> => {
    const url = typeof input === "string" ? input : (input as Request).url;
    let raw: string | undefined = (init.body as string | undefined) ?? undefined;
    if (raw === undefined && input instanceof Request) raw = await input.clone().text();
    let body: Record<string, unknown> | undefined;
    try {
      body = raw ? (JSON.parse(raw) as Record<string, unknown>) : undefined;
    } catch {
      body = undefined;
    }
    const req: CapturedRequest = { url, body };
    calls.push(req);
    return responder(req);
  };
  const model = new ChatInterfaze({
    apiKey: "test-key",
    maxRetries: 0,
    configuration: { fetch: fetchImpl as unknown as never },
    ...extraFields,
  });
  return { model, calls };
}

export function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
}

export function sseResponse(chunks: unknown[]): Response {
  const body = `${chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join("")}data: [DONE]\n\n`;
  return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
}

export function completion(content: unknown = "Hi!", extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: "req-test",
    object: "chat.completion",
    created: 1_700_000_000,
    model: "interfaze-beta",
    choices: [{ index: 0, message: { role: "assistant", content, refusal: null }, finish_reason: "stop", logprobs: null }],
    usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
    vcache: false,
    ...extra,
  };
}

export function chunk(delta: Record<string, unknown>, finishReason: string | null = null): Record<string, unknown> {
  return {
    id: "req-test",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "interfaze-beta",
    choices: [{ index: 0, delta: { role: "assistant", ...delta }, finish_reason: finishReason }],
  };
}

export function lastBody(calls: CapturedRequest[]): Record<string, unknown> {
  const body = calls.at(-1)?.body;
  if (!body) throw new Error("no request captured");
  return body;
}
