import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const WARNING = "Interfaze SDK is running in keyless mode. For higher rate limits, get an API key at https://interfaze.ai/dashboard.";

// Synthetic keys. No value here is a real credential.
const EXPLICIT_KEY = "sk-test-explicit";
const ENV_KEY = "sk-test-env";
const AUTHORIZATION = { authorization: "Bearer sk-test-header" };

// A fresh module copy per call resets the once-per-process flag.
const freshChatInterfaze = async (): Promise<typeof import("../src/index.js").ChatInterfaze> => {
  vi.resetModules();
  return (await import("../src/index.js")).ChatInterfaze;
};

let warn: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  vi.stubEnv("INTERFAZE_API_KEY", undefined);
  vi.stubEnv("OPENAI_API_KEY", undefined);
  vi.stubEnv("OPENAI_ADMIN_KEY", undefined);
  warn = vi.spyOn(console, "warn").mockImplementation(() => {});
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("the keyless warning", () => {
  it("warns once with the exact text for the first keyless model", async () => {
    const ChatInterfaze = await freshChatInterfaze();
    new ChatInterfaze();
    expect(warn.mock.calls).toEqual([[WARNING]]);
  });

  it("does not warn again for a second keyless model", async () => {
    const ChatInterfaze = await freshChatInterfaze();
    new ChatInterfaze();
    new ChatInterfaze();
    expect(warn).toHaveBeenCalledTimes(1);
  });

  it("warns for a keyless model built after a keyed one", async () => {
    const ChatInterfaze = await freshChatInterfaze();
    new ChatInterfaze({ apiKey: EXPLICIT_KEY });
    expect(warn).not.toHaveBeenCalled();
    new ChatInterfaze();
    expect(warn.mock.calls).toEqual([[WARNING]]);
  });

  it("does not warn for an explicit key", async () => {
    const ChatInterfaze = await freshChatInterfaze();
    new ChatInterfaze({ apiKey: EXPLICIT_KEY });
    expect(warn).not.toHaveBeenCalled();
  });

  it("does not warn for an env key", async () => {
    vi.stubEnv("INTERFAZE_API_KEY", ENV_KEY);
    const ChatInterfaze = await freshChatInterfaze();
    new ChatInterfaze();
    expect(warn).not.toHaveBeenCalled();
  });

  it("does not warn for a caller Authorization header", async () => {
    const ChatInterfaze = await freshChatInterfaze();
    new ChatInterfaze({ configuration: { defaultHeaders: AUTHORIZATION } });
    expect(warn).not.toHaveBeenCalled();
  });

  it("does not warn when a browser model throws", async () => {
    vi.stubGlobal("importScripts", () => {});
    const ChatInterfaze = await freshChatInterfaze();
    expect(() => new ChatInterfaze()).toThrow();
    expect(warn).not.toHaveBeenCalled();
  });
});
