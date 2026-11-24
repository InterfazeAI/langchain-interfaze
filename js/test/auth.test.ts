import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatInterfaze, ChatInterfazeFields } from "../src/index.js";
import { type CapturedRequest, chunk, completion, jsonResponse, mockChat, sseResponse } from "./helpers.js";

const ENV_KEY = "sk-env-synthetic";
const EXPLICIT_KEY = "sk-explicit-synthetic";
const OPENAI_KEY = "sk-openai-synthetic";

// Error bodies in the shape the API sends: a key it rejects, and a keyless caller over a daily limit.
const INVALID_KEY_BODY = {
  error: {
    message: "The API key provided is not valid or has been revoked",
    type: "permission_error",
    code: "insufficient_quota",
    request_id: "req-test",
  },
};
const KEYLESS_HINT = "Get a free API key at https://interfaze.ai";
const KEYLESS_LIMIT_BODY = {
  error: {
    message: `Keyless daily limit of 50 requests reached. ${KEYLESS_HINT} and send it as \`Authorization: Bearer <key>\`.`,
    type: "rate_limit_error",
    code: "keyless_daily_limit_exceeded",
    request_id: "req-test",
  },
};
const KEYLESS_LIMIT_HEADERS = { "retry-after": "3600", "x-ratelimit-limit": "50", "x-ratelimit-remaining": "0" };

const KEYLESS: Partial<ChatInterfazeFields> = { apiKey: undefined };

const errorResponse = (status: number, body: unknown, headers: Record<string, string> = {}): Response =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });

const drain = async (iterator: AsyncIterator<unknown>): Promise<void> => {
  const { done } = await iterator.next();
  return done ? undefined : drain(iterator);
};

type Call = (model: ChatInterfaze) => Promise<unknown>;

const invoke: Call = (model) => model.invoke("hi");
const stream: Call = (model) => model.stream("hi").then((chunks) => drain(chunks[Symbol.asyncIterator]()));

const CALLS: Array<[string, () => Response, Call]> = [
  ["invoke", () => jsonResponse(completion("Hi!")), invoke],
  ["stream", () => sseResponse([chunk({ content: "Hi!" }), chunk({}, "stop")]), stream],
];

const ERROR_CALLS: Array<[string, Call]> = [
  ["invoke", invoke],
  ["stream", stream],
];

const sentHeaders = async (respond: () => Response, call: Call, fields: Partial<ChatInterfazeFields>): Promise<Headers> => {
  const { model, calls } = mockChat(respond, fields);
  await call(model);
  const last = calls.at(-1);
  if (!last) throw new Error("no request captured");
  return last.headers;
};

const authorizations = (calls: CapturedRequest[]): Array<string | null> => calls.map((c) => c.headers.get("authorization"));

beforeEach(() => {
  vi.stubEnv("INTERFAZE_API_KEY", undefined);
  vi.stubEnv("OPENAI_API_KEY", undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("without a key", () => {
  it.each(CALLS)("%s sends no Authorization header", async (_label, respond, call) => {
    expect((await sentHeaders(respond, call, KEYLESS)).has("authorization")).toBe(false);
  });

  it.each(CALLS)("%s ignores OPENAI_API_KEY", async (_label, respond, call) => {
    vi.stubEnv("OPENAI_API_KEY", OPENAI_KEY);
    const headers = await sentHeaders(respond, call, KEYLESS);
    expect(headers.has("authorization")).toBe(false);
    headers.forEach((value) => expect(value).not.toContain(OPENAI_KEY));
  });

  it("keeps the headers the caller passes for one call", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion("Hi!")), KEYLESS);
    await model.invoke("hi", { options: { headers: { "x-trace": "1" } } });
    expect(calls.at(-1)?.headers.get("x-trace")).toBe("1");
    expect(calls.at(-1)?.headers.has("authorization")).toBe(false);
  });

  it.each(ERROR_CALLS)("%s surfaces a keyless 429 with the key hint", async (_label, call) => {
    const { model, calls } = mockChat(() => errorResponse(429, KEYLESS_LIMIT_BODY, KEYLESS_LIMIT_HEADERS), KEYLESS);
    await expect(call(model)).rejects.toMatchObject({
      status: 429,
      code: "keyless_daily_limit_exceeded",
      message: expect.stringContaining(KEYLESS_HINT),
    });
    expect(authorizations(calls)).toEqual([null]);
  });
});

describe("with a key", () => {
  it.each(CALLS)("%s sends INTERFAZE_API_KEY", async (_label, respond, call) => {
    vi.stubEnv("INTERFAZE_API_KEY", ENV_KEY);
    vi.stubEnv("OPENAI_API_KEY", OPENAI_KEY);
    expect((await sentHeaders(respond, call, KEYLESS)).get("authorization")).toBe(`Bearer ${ENV_KEY}`);
  });

  it.each(CALLS)("%s prefers an explicit key over INTERFAZE_API_KEY", async (_label, respond, call) => {
    vi.stubEnv("INTERFAZE_API_KEY", ENV_KEY);
    expect((await sentHeaders(respond, call, { apiKey: EXPLICIT_KEY })).get("authorization")).toBe(`Bearer ${EXPLICIT_KEY}`);
  });

  it.each(ERROR_CALLS)("%s surfaces a 401 without a keyless retry", async (_label, call) => {
    const { model, calls } = mockChat(() => errorResponse(401, INVALID_KEY_BODY), { apiKey: EXPLICIT_KEY, maxRetries: 2 });
    await expect(call(model)).rejects.toMatchObject({ status: 401, code: "insufficient_quota" });
    expect(authorizations(calls)).toEqual([`Bearer ${EXPLICIT_KEY}`]);
  });
});
