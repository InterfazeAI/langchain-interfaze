import { AIMessage } from "@langchain/core/messages";
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { ChatInterfaze } from "../src/index.js";
import { VERSION } from "../src/version.js";
import { chunk, completion, jsonResponse, mockChat, sseResponse } from "./helpers.js";

describe("provider identity", () => {
  const model = new ChatInterfaze({ apiKey: "t" });

  it("reports interfaze, not openai", () => {
    expect(model._llmType()).toBe("interfaze");
    expect(model.getName()).toBe("ChatInterfaze");
    expect(model.lc_namespace).toEqual(["langchain", "chat_models", "interfaze"]);
    expect(model.lc_secrets).toEqual({ apiKey: "INTERFAZE_API_KEY" });
  });

  it("tags langsmith params with the interfaze provider", () => {
    const params = model.getLsParams({} as never);
    expect(params.ls_provider).toBe("interfaze");
    expect(params.ls_model_name).toBe("interfaze-beta");
    expect(params.ls_model_type).toBe("chat");
  });

  it("records its own package version alongside core's", () => {
    const versions = (model as unknown as { metadata?: { versions?: Record<string, string> } }).metadata?.versions ?? {};
    expect(versions["@interfaze-ai/langchain"]).toBe(VERSION);
    expect(versions["@langchain/core"]).toBeTypeOf("string");
  });

  it("keeps VERSION in sync with package.json", () => {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as { version: string };
    expect(VERSION).toBe(pkg.version);
  });

  it("stamps model_provider on invoke responses", async () => {
    const { model: m } = mockChat(() => jsonResponse(completion("hi")));
    const res = (await m.invoke("hi")) as AIMessage;
    expect(res.response_metadata.model_provider).toBe("interfaze");
  });

  it("stamps model_provider on streamed chunks", async () => {
    const { model: m } = mockChat(() => sseResponse([chunk({ content: "hi" }), chunk({}, "stop")]));
    const providers: unknown[] = [];
    for await (const c of await m.stream("hi")) providers.push(c.response_metadata.model_provider);
    expect(providers.every((p) => p === "interfaze")).toBe(true);
  });
});
