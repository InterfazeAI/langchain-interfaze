import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { InterfazeError } from "interfaze";
import { ChatInterfaze, type ChatInterfazeFields } from "../src/index.js";
import { type CapturedRequest, chunk, completion, jsonResponse, recordingFetch, sseResponse } from "./helpers.js";

// Synthetic keys. No value here is a real credential.
const STAND_IN = "interfaze-keyless";
const ENV_KEY = "sk-test-env";
const EXPLICIT_KEY = "sk-test-explicit";
const OPENAI_KEY = "sk-test-openai-env";
const OPENAI_FIELD_KEY = "sk-test-openai-field";
const CONFIG_KEY = "sk-test-config";
const FUNCTION_KEY = "sk-test-function";
const HEADER_KEY = "sk-test-header";
const NEW_KEY = "sk-test-new";
const ADMIN_KEY = "sk-test-admin";
const bearer = (key: string): string => `Bearer ${key}`;

// Error bodies in the shape the API sends.
const REQUEST_ID = "req-test";
const HINT = "Get a free API key at https://interfaze.ai and send it as `Authorization: Bearer YOUR_API_KEY`.";
const INVALID_KEY_BODY = {
  error: {
    message: "The API key provided is not valid",
    type: "permission_error",
    code: "insufficient_quota",
    request_id: REQUEST_ID,
  },
};
const DAILY_LIMIT_BODY = {
  error: {
    message: `Keyless daily limit of 1000 requests reached. ${HINT}`,
    type: "rate_limit_error",
    code: "keyless_daily_limit_exceeded",
    request_id: REQUEST_ID,
  },
};
const IN_FLIGHT_BODY = {
  error: {
    message: `Too many keyless requests in progress (limit 5). ${HINT}`,
    type: "rate_limit_error",
    code: "keyless_inflight_limit_exceeded",
    request_id: REQUEST_ID,
  },
};

const DAILY_RETRY_AFTER_S = 3600;
const DAILY_HEADERS = { "retry-after": String(DAILY_RETRY_AFTER_S) };
const IN_FLIGHT_RETRY_AFTER_S = 5;
const IN_FLIGHT_HEADERS = { "retry-after": String(IN_FLIGHT_RETRY_AFTER_S) };
const IN_FLIGHT_MAX_RETRIES = 1;
const KEYED_MAX_RETRIES = 2;
const NO_WAIT_LIMIT_MS = 2000;
const PAST_RETRY_AFTER_MS = 60_000;
const CLIENT_TIMEOUT_MS = 5000;

const errorResponse = (status: number, body: unknown, headers: Record<string, string> = {}): Response =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });

// One responder for every path: models, a stream, or a completion.
const respond = (req: CapturedRequest): Response => {
  if (req.url.endsWith("/models")) {
    return jsonResponse({ object: "list", data: [{ id: "interfaze-beta", object: "model", created: 1, owned_by: "interfaze" }] });
  }
  if (req.body?.stream) return sseResponse([chunk({ content: "Hi!" }), chunk({}, "stop")]);
  return jsonResponse(completion("Hi!"));
};

const drain = async (iterator: AsyncIterator<unknown>): Promise<void> => {
  const { done } = await iterator.next();
  return done ? undefined : drain(iterator);
};

type Fields = Partial<ChatInterfazeFields>;
type Call = (model: ChatInterfaze) => Promise<unknown>;

const build = (responder: (req: CapturedRequest) => Response, fields: Fields = {}): { model: ChatInterfaze; calls: CapturedRequest[] } => {
  const { fetch, calls } = recordingFetch(responder);
  const model = new ChatInterfaze({ ...fields, configuration: { ...fields.configuration, fetch } });
  return { model, calls };
};

const rawClient = (model: ChatInterfaze): ChatInterfaze["client"] => {
  model._getClientOptions(undefined);
  return model.client;
};

const invoke: Call = (model) => model.invoke("hi");

const PATHS: Array<[string, Call]> = [
  ["invoke", invoke],
  ["stream", (model) => model.stream("hi").then((chunks) => drain(chunks[Symbol.asyncIterator]()))],
  ["client.models.list", (model) => rawClient(model).models.list()],
  ["client.withOptions", (model) => rawClient(model).withOptions({ timeout: CLIENT_TIMEOUT_MS }).models.list()],
];

interface Scenario {
  name: string;
  env?: Record<string, string>;
  fields?: Fields;
  expected: string | null;
}

const SCENARIOS: Scenario[] = [
  { name: "only OPENAI_API_KEY", env: { OPENAI_API_KEY: OPENAI_KEY }, expected: null },
  { name: "trimmed env key", env: { INTERFAZE_API_KEY: ` ${ENV_KEY}\n` }, expected: bearer(ENV_KEY) },
  {
    name: "trimmed explicit key over env key",
    env: { INTERFAZE_API_KEY: ENV_KEY },
    fields: { apiKey: ` ${EXPLICIT_KEY}\n` },
    expected: bearer(EXPLICIT_KEY),
  },
  { name: "blank explicit key with env key", env: { INTERFAZE_API_KEY: ENV_KEY }, fields: { apiKey: " \n" }, expected: bearer(ENV_KEY) },
  { name: "blank env key", env: { INTERFAZE_API_KEY: " \n" }, expected: null },
  { name: "function key", fields: { configuration: { apiKey: async () => FUNCTION_KEY } }, expected: bearer(FUNCTION_KEY) },
  {
    name: "openAIApiKey over env key",
    env: { INTERFAZE_API_KEY: ENV_KEY },
    fields: { openAIApiKey: OPENAI_FIELD_KEY },
    expected: bearer(OPENAI_FIELD_KEY),
  },
  {
    name: "configuration.apiKey over env key",
    env: { INTERFAZE_API_KEY: ENV_KEY },
    fields: { configuration: { apiKey: CONFIG_KEY } },
    expected: bearer(CONFIG_KEY),
  },
  {
    name: "caller Authorization header without a key",
    fields: { configuration: { defaultHeaders: { Authorization: bearer(HEADER_KEY) } } },
    expected: bearer(HEADER_KEY),
  },
  {
    name: "caller Authorization header over an explicit key",
    fields: { apiKey: EXPLICIT_KEY, configuration: { defaultHeaders: { AUTHORIZATION: bearer(HEADER_KEY) } } },
    expected: bearer(HEADER_KEY),
  },
  { name: "blank caller Authorization header without a key", fields: { configuration: { defaultHeaders: { Authorization: "" } } }, expected: null },
];

beforeEach(() => {
  vi.stubEnv("INTERFAZE_API_KEY", undefined);
  vi.stubEnv("OPENAI_API_KEY", undefined);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.useRealTimers();
});

const authorizations = (calls: CapturedRequest[]): Array<string | null> => calls.map((c) => c.headers.get("authorization"));

const expectNoStandIn = (calls: CapturedRequest[]): void =>
  calls.forEach((c) => c.headers.forEach((value) => expect(value).not.toMatch(new RegExp(`${STAND_IN}|${OPENAI_KEY}`))));

describe("a model with no key", () => {
  it.each(PATHS)("%s sends no Authorization header", async (_label, call) => {
    vi.stubEnv("OPENAI_API_KEY", OPENAI_KEY);
    const { model, calls } = build(respond);
    await call(model);
    expect(calls.length).toBeGreaterThan(0);
    expect(authorizations(calls)).toEqual(calls.map(() => null));
    expectNoStandIn(calls);
  });

  it("sends the other headers the caller sets for one call", async () => {
    const { model, calls } = build(respond);
    await model.invoke("hi", { options: { headers: { "x-trace": "1" } } });
    expect(calls.at(-1)?.headers.get("x-trace")).toBe("1");
    expect(calls.at(-1)?.headers.has("authorization")).toBe(false);
  });

  it("uses the global fetch when the caller passes none", async () => {
    const { fetch, calls } = recordingFetch(respond);
    vi.stubGlobal("fetch", fetch);
    await new ChatInterfaze().invoke("hi");
    expect(authorizations(calls)).toEqual([null]);
  });

  it("sends a new key given to a raw client through withOptions", async () => {
    const { model, calls } = build(respond);
    await rawClient(model).withOptions({ apiKey: NEW_KEY }).models.list();
    expect(authorizations(calls)).toEqual([bearer(NEW_KEY)]);
  });

  it("sends no OPENAI_ADMIN_KEY from the environment on an admin endpoint", async () => {
    vi.stubEnv("OPENAI_ADMIN_KEY", ADMIN_KEY);
    const { model, calls } = build(respond);
    await expect(rawClient(model).admin.organization.usage.completions({ start_time: 1 })).rejects.toThrow();
    expect(calls).toEqual([]);
  });

  it("stays keyless in a runtime with no process and no document", async () => {
    vi.stubGlobal("process", undefined);
    const { model, calls } = build(respond);
    await model.invoke("hi");
    expect(authorizations(calls)).toEqual([null]);
  });
});

describe.each(SCENARIOS)("$name", ({ env, fields, expected }) => {
  it("sends the expected Authorization header", async () => {
    Object.entries(env ?? {}).forEach(([name, value]) => vi.stubEnv(name, value));
    const { model, calls } = build(respond, fields);
    await invoke(model);
    expect(authorizations(calls)).toEqual([expected]);
    expectNoStandIn(calls);
  });
});

describe("errors", () => {
  it("surfaces a 401 after one request with the status, code and body", async () => {
    const { model, calls } = build(() => errorResponse(401, INVALID_KEY_BODY), { apiKey: EXPLICIT_KEY, maxRetries: KEYED_MAX_RETRIES });
    await expect(invoke(model)).rejects.toMatchObject({
      status: 401,
      code: "insufficient_quota",
      type: "permission_error",
      message: expect.stringContaining("The API key provided is not valid"),
    });
    expect(authorizations(calls)).toEqual([bearer(EXPLICIT_KEY)]);
  });

  it("surfaces a keyless daily-limit 429 after one request with no wait", async () => {
    const { model, calls } = build(() => errorResponse(429, DAILY_LIMIT_BODY, DAILY_HEADERS));
    const started = Date.now();
    await expect(invoke(model)).rejects.toMatchObject({
      status: 429,
      code: "keyless_daily_limit_exceeded",
      type: "rate_limit_error",
      message: expect.stringContaining(HINT),
    });
    expect(Date.now() - started).toBeLessThan(NO_WAIT_LIMIT_MS);
    expect(authorizations(calls)).toEqual([null]);
  });

  it("lets LangChain's caller retry a keyless in-flight 429", async () => {
    vi.useFakeTimers();
    const { model, calls } = build(() => errorResponse(429, IN_FLIGHT_BODY, IN_FLIGHT_HEADERS), { maxRetries: IN_FLIGHT_MAX_RETRIES });
    const rejection = expect(invoke(model)).rejects.toMatchObject({
      status: 429,
      code: "keyless_inflight_limit_exceeded",
      message: expect.stringContaining(HINT),
    });
    await vi.advanceTimersByTimeAsync(PAST_RETRY_AFTER_MS);
    await rejection;
    expect(authorizations(calls)).toEqual([null, null]);
  });
});

describe("a browser", () => {
  it("throws InterfazeError without a usable key, with and without a process polyfill", () => {
    vi.stubGlobal("document", {});
    expect(() => new ChatInterfaze()).toThrow(InterfazeError);
    expect(() => new ChatInterfaze({ apiKey: " \n" })).toThrow(InterfazeError);
    vi.stubGlobal("process", { env: {} });
    expect(() => new ChatInterfaze()).toThrow(InterfazeError);
  });

  it.each<[string, Record<string, string>, Fields]>([
    ["an explicit key", {}, { apiKey: EXPLICIT_KEY }],
    ["an env key", { INTERFAZE_API_KEY: ENV_KEY }, {}],
    ["a caller Authorization header", {}, { configuration: { defaultHeaders: { authorization: bearer(HEADER_KEY) } } }],
  ])("builds with %s", (_label, env, fields) => {
    vi.stubGlobal("document", {});
    Object.entries(env).forEach(([name, value]) => vi.stubEnv(name, value));
    expect(() => new ChatInterfaze(fields)).not.toThrow();
  });
});
