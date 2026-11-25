import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { InterfazeError } from "interfaze";
import { z } from "zod";
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
const REQUEST_KEY = "sk-test-request";
const NEW_KEY = "sk-test-new";
const ADMIN_KEY = "sk-test-admin";
const bearer = (key: string): string => `Bearer ${key}`;

const BLANK_KEYS = ["", " ", " \n"];

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
const DAILY_LIMIT = 1000;
const dailyHeaders = (): Record<string, string> => ({
  "retry-after": String(DAILY_RETRY_AFTER_S),
  "x-ratelimit-limit": String(DAILY_LIMIT),
  "x-ratelimit-remaining": "0",
  "x-ratelimit-reset": String(Math.floor(Date.now() / 1000) + DAILY_RETRY_AFTER_S),
});
const IN_FLIGHT_RETRY_AFTER_S = 5;
const IN_FLIGHT_HEADERS = { "retry-after": String(IN_FLIGHT_RETRY_AFTER_S) };
const IN_FLIGHT_MAX_RETRIES = 1;
const KEYED_MAX_RETRIES = 2;
const NO_WAIT_LIMIT_MS = 2000;
const PAST_RETRY_AFTER_MS = 60_000;
const CLIENT_TIMEOUT_MS = 5000;

const errorResponse = (status: number, body: unknown, headers: Record<string, string> = {}): Response =>
  new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json", ...headers } });

// One responder for every path: models, moderations, a stream, or a completion.
const respond = (req: CapturedRequest): Response => {
  if (req.url.endsWith("/models")) {
    return jsonResponse({ object: "list", data: [{ id: "interfaze-beta", object: "model", created: 1, owned_by: "interfaze" }] });
  }
  if (req.url.endsWith("/moderations")) return jsonResponse({ id: "modr-test", model: "omni-moderation-latest", results: [] });
  if (req.body?.stream) return sseResponse([chunk({ content: "Hi!" }), chunk({}, "stop")]);
  return jsonResponse(completion('{"a":1}'));
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

const schema = z.object({ a: z.number() });

const invoke: Call = (model) => model.invoke("hi");
const stream: Call = (model) => model.stream("hi").then((chunks) => drain(chunks[Symbol.asyncIterator]()));

const PATHS: Array<[string, Call]> = [
  ["invoke", invoke],
  ["stream", stream],
  ["batch", (model) => model.batch(["a", "b"])],
  ["withStructuredOutput", (model) => model.withStructuredOutput(schema, { method: "jsonSchema" }).invoke("hi")],
  ["moderateContent", (model) => model.moderateContent("hi")],
  ["client.models.list", (model) => rawClient(model).models.list()],
  [
    "client.chat.completions.create",
    (model) => rawClient(model).chat.completions.create({ model: "interfaze-beta", messages: [{ role: "user", content: "hi" }] }),
  ],
  ["client.withOptions", (model) => rawClient(model).withOptions({ timeout: CLIENT_TIMEOUT_MS }).models.list()],
];

const ERROR_PATHS: Array<[string, Call]> = [
  ["invoke", invoke],
  ["stream", stream],
];

interface Scenario {
  name: string;
  env?: Record<string, string>;
  fields?: Fields;
  expected: string | null;
}

const functionKey = (): Fields["configuration"] => ({ apiKey: async () => FUNCTION_KEY });

const SCENARIOS: Scenario[] = [
  { name: "no key", expected: null },
  { name: "only OPENAI_API_KEY", env: { OPENAI_API_KEY: OPENAI_KEY }, expected: null },
  { name: "env key", env: { INTERFAZE_API_KEY: ENV_KEY }, expected: bearer(ENV_KEY) },
  { name: "explicit key over env key", env: { INTERFAZE_API_KEY: ENV_KEY }, fields: { apiKey: EXPLICIT_KEY }, expected: bearer(EXPLICIT_KEY) },
  ...BLANK_KEYS.map((apiKey) => ({ name: `blank explicit ${JSON.stringify(apiKey)}`, fields: { apiKey }, expected: null })),
  ...BLANK_KEYS.map((apiKey) => ({
    name: `blank explicit ${JSON.stringify(apiKey)} with env key`,
    env: { INTERFAZE_API_KEY: ENV_KEY },
    fields: { apiKey },
    expected: bearer(ENV_KEY),
  })),
  ...BLANK_KEYS.map((value) => ({ name: `blank env ${JSON.stringify(value)}`, env: { INTERFAZE_API_KEY: value }, expected: null })),
  { name: "function key", fields: { configuration: functionKey() }, expected: bearer(FUNCTION_KEY) },
  { name: "openAIApiKey", fields: { openAIApiKey: OPENAI_FIELD_KEY }, expected: bearer(OPENAI_FIELD_KEY) },
  {
    name: "openAIApiKey over env key",
    env: { INTERFAZE_API_KEY: ENV_KEY },
    fields: { openAIApiKey: OPENAI_FIELD_KEY },
    expected: bearer(OPENAI_FIELD_KEY),
  },
  { name: "configuration.apiKey", fields: { configuration: { apiKey: CONFIG_KEY } }, expected: bearer(CONFIG_KEY) },
  {
    name: "configuration.apiKey over env key",
    env: { INTERFAZE_API_KEY: ENV_KEY },
    fields: { configuration: { apiKey: CONFIG_KEY } },
    expected: bearer(CONFIG_KEY),
  },
  { name: "blank apiKey with configuration.apiKey", fields: { apiKey: " ", configuration: { apiKey: CONFIG_KEY } }, expected: bearer(CONFIG_KEY) },
  { name: "blank configuration.apiKey", fields: { configuration: { apiKey: " \n" } }, expected: null },
  {
    name: "blank configuration.apiKey with env key",
    env: { INTERFAZE_API_KEY: ENV_KEY },
    fields: { configuration: { apiKey: " " } },
    expected: bearer(ENV_KEY),
  },
  {
    name: "caller Authorization header without a key",
    fields: { configuration: { defaultHeaders: { Authorization: bearer(HEADER_KEY) } } },
    expected: bearer(HEADER_KEY),
  },
  {
    name: "lowercase caller authorization header without a key",
    fields: { configuration: { defaultHeaders: { authorization: bearer(HEADER_KEY) } } },
    expected: bearer(HEADER_KEY),
  },
  {
    name: "Headers instance with Authorization without a key",
    fields: { configuration: { defaultHeaders: new Headers({ Authorization: bearer(HEADER_KEY) }) } },
    expected: bearer(HEADER_KEY),
  },
  ...BLANK_KEYS.map((value) => ({
    name: `blank caller Authorization header ${JSON.stringify(value)} without a key`,
    fields: { configuration: { defaultHeaders: { Authorization: value } } },
    expected: null,
  })),
  ...BLANK_KEYS.map((value) => ({
    name: `blank caller Authorization header ${JSON.stringify(value)} with an explicit key`,
    fields: { apiKey: EXPLICIT_KEY, configuration: { defaultHeaders: { Authorization: value } } },
    expected: bearer(EXPLICIT_KEY),
  })),
  {
    name: "caller Authorization header over an explicit key",
    fields: { apiKey: EXPLICIT_KEY, configuration: { defaultHeaders: { AUTHORIZATION: bearer(HEADER_KEY) } } },
    expected: bearer(HEADER_KEY),
  },
  {
    name: "caller Authorization header over an env key",
    env: { INTERFAZE_API_KEY: ENV_KEY },
    fields: { configuration: { defaultHeaders: { Authorization: bearer(HEADER_KEY) } } },
    expected: bearer(HEADER_KEY),
  },
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

describe.each(SCENARIOS)("$name", ({ env, fields, expected }) => {
  it.each(PATHS)("%s sends the expected Authorization header on every request", async (_label, call) => {
    Object.entries(env ?? {}).forEach(([name, value]) => vi.stubEnv(name, value));
    const { model, calls } = build(respond, fields);
    await call(model);
    expect(calls.length).toBeGreaterThan(0);
    expect(authorizations(calls)).toEqual(calls.map(() => expected));
    calls.forEach((c) => c.headers.forEach((value) => expect(value).not.toMatch(new RegExp(`${STAND_IN}|${OPENAI_KEY}`))));
  });
});

describe("the key a model keeps", () => {
  it("passes a function key through untouched", () => {
    const apiKey = async (): Promise<string> => FUNCTION_KEY;
    const model = new ChatInterfaze({ configuration: { apiKey } });
    expect(model.clientConfig.apiKey).toBe(apiKey);
    expect(model.apiKey).toBe(apiKey);
  });

  it("trims a string key", () => {
    const explicit = new ChatInterfaze({ apiKey: ` ${EXPLICIT_KEY}\n` });
    expect(explicit.clientConfig.apiKey).toBe(EXPLICIT_KEY);
    expect(explicit.apiKey).toBe(EXPLICIT_KEY);
    expect(new ChatInterfaze({ configuration: { apiKey: ` ${CONFIG_KEY} ` } }).clientConfig.apiKey).toBe(CONFIG_KEY);
    vi.stubEnv("INTERFAZE_API_KEY", ` ${ENV_KEY}\n`);
    expect(new ChatInterfaze().clientConfig.apiKey).toBe(ENV_KEY);
  });

  it("leaves the caller's fetch alone when a key exists", () => {
    const { fetch } = recordingFetch(respond);
    expect(new ChatInterfaze({ apiKey: EXPLICIT_KEY, configuration: { fetch } }).clientConfig.fetch).toBe(fetch);
  });
});

describe("a request header the caller sets for one call", () => {
  const headers = { Authorization: bearer(REQUEST_KEY) };
  const HEADER_PATHS: Array<[string, Call]> = [
    ["invoke", (model) => model.invoke("hi", { options: { headers } })],
    ["moderateContent", (model) => model.moderateContent("hi", { options: { headers } })],
    ["client.models.list", (model) => rawClient(model).models.list({ headers })],
  ];

  it.each(HEADER_PATHS)("%s sends it without a key", async (_label, call) => {
    const { model, calls } = build(respond);
    await call(model);
    expect(authorizations(calls)).toEqual([bearer(REQUEST_KEY)]);
  });

  it("keeps the other headers the caller passes", async () => {
    const { model, calls } = build(respond);
    await model.invoke("hi", { options: { headers: { "x-trace": "1" } } });
    expect(calls.at(-1)?.headers.get("x-trace")).toBe("1");
    expect(calls.at(-1)?.headers.has("authorization")).toBe(false);
  });
});

describe("a raw client", () => {
  it("sends a new key given through withOptions", async () => {
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

  it("uses the global fetch when the caller passes none", async () => {
    const { fetch, calls } = recordingFetch(respond);
    vi.stubGlobal("fetch", fetch);
    await new ChatInterfaze().invoke("hi");
    expect(authorizations(calls)).toEqual([null]);
  });
});

describe("errors", () => {
  it.each(ERROR_PATHS)("%s surfaces a 401 after one request with the status, code and body", async (_label, call) => {
    const { model, calls } = build(() => errorResponse(401, INVALID_KEY_BODY), { apiKey: EXPLICIT_KEY, maxRetries: KEYED_MAX_RETRIES });
    await expect(call(model)).rejects.toMatchObject({
      status: 401,
      code: "insufficient_quota",
      type: "permission_error",
      message: expect.stringContaining("The API key provided is not valid"),
    });
    expect(authorizations(calls)).toEqual([bearer(EXPLICIT_KEY)]);
  });

  it.each(ERROR_PATHS)("%s does not retry a 401 for a caller header without a key", async (_label, call) => {
    const { model, calls } = build(() => errorResponse(401, INVALID_KEY_BODY), {
      maxRetries: KEYED_MAX_RETRIES,
      configuration: { defaultHeaders: { Authorization: bearer(HEADER_KEY) } },
    });
    await expect(call(model)).rejects.toMatchObject({ status: 401, code: "insufficient_quota" });
    expect(authorizations(calls)).toEqual([bearer(HEADER_KEY)]);
  });

  it.each(ERROR_PATHS)("%s surfaces a keyless daily-limit 429 after one request with no wait", async (_label, call) => {
    const { model, calls } = build(() => errorResponse(429, DAILY_LIMIT_BODY, dailyHeaders()));
    const started = Date.now();
    await expect(call(model)).rejects.toMatchObject({
      status: 429,
      code: "keyless_daily_limit_exceeded",
      type: "rate_limit_error",
      message: expect.stringContaining(HINT),
    });
    expect(Date.now() - started).toBeLessThan(NO_WAIT_LIMIT_MS);
    expect(authorizations(calls)).toEqual([null]);
  });

  it.each(ERROR_PATHS)("%s keeps the retry that LangChain's caller makes on a keyless in-flight 429", async (_label, call) => {
    vi.useFakeTimers();
    const { model, calls } = build(() => errorResponse(429, IN_FLIGHT_BODY, IN_FLIGHT_HEADERS), { maxRetries: IN_FLIGHT_MAX_RETRIES });
    const rejection = expect(call(model)).rejects.toMatchObject({
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
  it("throws InterfazeError when no key exists, also with a process polyfill", () => {
    vi.stubGlobal("document", {});
    vi.stubGlobal("process", { env: {} });
    expect(() => new ChatInterfaze()).toThrow(InterfazeError);
  });

  it("throws InterfazeError when no key exists and process is the real one", () => {
    vi.stubGlobal("document", {});
    expect(() => new ChatInterfaze()).toThrow(InterfazeError);
  });

  it.each(BLANK_KEYS)("throws for the blank key %j", (apiKey) => {
    vi.stubGlobal("document", {});
    expect(() => new ChatInterfaze({ apiKey })).toThrow(InterfazeError);
  });

  it("sends no request when it throws", () => {
    vi.stubGlobal("document", {});
    const { fetch, calls } = recordingFetch(respond);
    expect(() => new ChatInterfaze({ configuration: { fetch } })).toThrow(InterfazeError);
    expect(calls).toEqual([]);
  });

  it.each<[string, Fields]>([
    ["an explicit key", { apiKey: EXPLICIT_KEY }],
    ["openAIApiKey", { openAIApiKey: OPENAI_FIELD_KEY }],
    ["configuration.apiKey", { configuration: { apiKey: CONFIG_KEY } }],
    ["a function key", { configuration: functionKey() }],
    ["a caller Authorization header", { configuration: { defaultHeaders: { authorization: bearer(HEADER_KEY) } } }],
  ])("builds with %s", (_label, fields) => {
    vi.stubGlobal("document", {});
    expect(() => new ChatInterfaze(fields)).not.toThrow();
  });

  it("builds with an env key", () => {
    vi.stubGlobal("document", {});
    vi.stubEnv("INTERFAZE_API_KEY", ENV_KEY);
    expect(() => new ChatInterfaze()).not.toThrow();
  });
});

describe("a runtime with no process and no document", () => {
  it("stays keyless", async () => {
    vi.stubGlobal("process", undefined);
    const { model, calls } = build(respond);
    await model.invoke("hi");
    expect(authorizations(calls)).toEqual([null]);
  });
});
