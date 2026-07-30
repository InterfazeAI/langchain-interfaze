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
    // clientConfig carries the resolved baseURL
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

  it("injects the precontext field into the request body", async () => {
    const pc = [{ name: "ocr", result: { extracted_text: "y" } }];
    const { model, calls } = mockChat(() => jsonResponse(completion("Hi!")), { precontext: pc });
    await model.invoke("hi");
    expect(lastBody(calls).precontext).toEqual(pc);
  });

  it("omits precontext when not set", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion("Hi!")));
    await model.invoke("hi");
    expect("precontext" in lastBody(calls)).toBe(false);
  });
});
