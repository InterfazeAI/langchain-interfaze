import { afterEach, describe, expect, it } from "vitest";
import { INTERFAZE_BASE_URL, INTERFAZE_MODEL, InterfazeError } from "interfaze";
import { ChatInterfaze } from "../src/index.js";
import { completion, jsonResponse, lastBody, mockChat } from "./helpers.js";

const KEY = "INTERFAZE_API_KEY";
afterEach(() => {
  delete process.env[KEY];
});

describe("ChatInterfaze constructor", () => {
  it("defaults baseURL and model to Interfaze", () => {
    const model = new ChatInterfaze({ apiKey: "t" });
    expect(model.model).toBe(INTERFAZE_MODEL);
    expect((model as unknown as { clientConfig: { baseURL?: string } }).clientConfig.baseURL).toBe(INTERFAZE_BASE_URL);
  });

  it("lets base url and model be overridden", () => {
    const model = new ChatInterfaze({ apiKey: "t", model: "other-model", configuration: { baseURL: "https://example.com/v1" } });
    expect(model.model).toBe("other-model");
    expect((model as unknown as { clientConfig: { baseURL?: string } }).clientConfig.baseURL).toBe("https://example.com/v1");
  });

  it("throws InterfazeError when no api key is present", () => {
    delete process.env[KEY];
    expect(() => new ChatInterfaze()).toThrow(InterfazeError);
    expect(() => new ChatInterfaze()).toThrow(/Missing API key/);
  });

  it("reads the api key from INTERFAZE_API_KEY", () => {
    process.env[KEY] = "env-key";
    expect(() => new ChatInterfaze()).not.toThrow();
  });

  it("is not lc-serializable (closes the real langchain-core serialization gate)", () => {
    const model = new ChatInterfaze({ apiKey: "t" });
    expect(model.lc_serializable).toBe(false);
  });

  it("defaults to a long timeout but respects an override", () => {
    expect((new ChatInterfaze({ apiKey: "t" }) as unknown as { timeout?: number }).timeout).toBe(900_000);
    expect((new ChatInterfaze({ apiKey: "t", timeout: 30_000 }) as unknown as { timeout?: number }).timeout).toBe(30_000);
  });

  it("maps the interfaze control options onto request headers", () => {
    const model = new ChatInterfaze({
      apiKey: "t",
      showAdditionalInfo: true,
      bypassMoA: true,
      bypassCache: true,
      configuration: { defaultHeaders: { "x-custom": "1" } },
    });
    const headers = (model as unknown as { clientConfig: { defaultHeaders?: Record<string, string> } }).clientConfig.defaultHeaders;
    expect(headers).toEqual({
      "x-custom": "1",
      "x-show-additional-info": "true",
      "x-interfaze-bypass-moa": "true",
      "x-interfaze-bypass-cache": "true",
    });
  });

  // defaultHeaders is HeadersLike; spreading it loses non-plain-object shapes.
  it.each([
    ["plain object", { "x-tenant": "acme" } as never],
    ["Headers instance", new Headers({ "x-tenant": "acme" }) as never],
    ["tuple array", [["x-tenant", "acme"]] as never],
  ])("keeps a caller header passed as a %s", (_label, defaultHeaders) => {
    const model = new ChatInterfaze({ apiKey: "t", bypassCache: true, configuration: { defaultHeaders } });
    const headers = (model as unknown as { clientConfig: { defaultHeaders?: Record<string, string> } }).clientConfig.defaultHeaders;
    expect(headers?.["x-tenant"]).toBe("acme");
    expect(headers?.["x-interfaze-bypass-cache"]).toBe("true");
  });

  it("sends no control headers by default", () => {
    const model = new ChatInterfaze({ apiKey: "t" });
    expect((model as unknown as { clientConfig: { defaultHeaders?: unknown } }).clientConfig.defaultHeaders).toBeUndefined();
  });

  // @langchain/openai drops reasoning params for models its heuristic doesn't
  // recognize (/^o\d/, gpt-5*), so interfaze-beta loses them without our override.
  it.each([
    ["call option", async (m: ChatInterfaze) => m.invoke("hi", { reasoningEffort: "high" })],
    ["withConfig", async (m: ChatInterfaze) => m.withConfig({ reasoningEffort: "high" } as never).invoke("hi")],
  ])("forwards reasoning_effort via %s", async (_label, run) => {
    const { model, calls } = mockChat(() => jsonResponse(completion("Hi!")));
    await run(model);
    expect(lastBody(calls).reasoning_effort).toBe("high");
  });

  it("forwards a constructor reasoningEffort, including interfaze-only values", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion("Hi!")), { reasoningEffort: "on" });
    await model.invoke("hi");
    expect(lastBody(calls).reasoning_effort).toBe("on");
  });

  it("forwards a constructor reasoning.effort", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion("Hi!")), { reasoning: { effort: "high" } } as never);
    await model.invoke("hi");
    expect(lastBody(calls).reasoning_effort).toBe("high");
  });

  // Within one call site, `reasoning.effort` wins — matching upstream's own ordering.
  it("gives reasoning.effort precedence over reasoningEffort per call", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion("Hi!")));
    await model.invoke("hi", { reasoning: { effort: "low" }, reasoningEffort: "high" } as never);
    expect(lastBody(calls).reasoning_effort).toBe("low");
  });

  // ...but a per-call value always beats the constructor, in either form.
  // `.withConfig({reasoningEffort})` is the documented per-chain override.
  it.each([
    ["reasoning.effort", { reasoning: { effort: "low" } }],
    ["reasoningEffort", { reasoningEffort: "low" }],
  ])("lets a per-call effort override constructor %s", async (_label, ctor) => {
    const { model, calls } = mockChat(() => jsonResponse(completion("Hi!")), ctor as never);
    await model.invoke("hi", { reasoningEffort: "high" } as never);
    expect(lastBody(calls).reasoning_effort).toBe("high");

    await model.withConfig({ reasoningEffort: "minimal" } as never).invoke("hi");
    expect(lastBody(calls).reasoning_effort).toBe("minimal");
  });

  it("omits reasoning_effort when unset", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion("Hi!")));
    await model.invoke("hi");
    expect("reasoning_effort" in lastBody(calls)).toBe(false);
  });

  it("actually puts the control headers on the wire", async () => {
    let seen: Headers | undefined;
    const model = new ChatInterfaze({
      apiKey: "t",
      bypassCache: true,
      maxRetries: 0,
      configuration: {
        fetch: (async (input: unknown, init: RequestInit = {}) => {
          seen = new Headers((init.headers ?? (input as Request).headers) as HeadersInit);
          return jsonResponse(completion("Hi!"));
        }) as unknown as never,
      },
    });
    await model.invoke("hi");
    expect(seen?.get("x-interfaze-bypass-cache")).toBe("true");
  });
});

describe("header values", () => {
  it("keeps a scalar header value that normalizeHeaders would drop", () => {
    const model = new ChatInterfaze({ apiKey: "k", configuration: { defaultHeaders: { "X-Retries": 3, "x-debug": true } as never } });
    expect(model.clientConfig.defaultHeaders).toMatchObject({ "x-retries": "3", "x-debug": "true" });
  });
});
