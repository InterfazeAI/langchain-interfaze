import { describe, expect, it } from "vitest";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatInterfaze } from "../src/index.js";
import { completion, jsonResponse, mockChat, sseResponse, chunk } from "./helpers.js";

describe("ChatInterfaze conforms to the standard chat-model surface", () => {
  it("invoke returns an AIMessage with string content", async () => {
    const { model } = mockChat(() => jsonResponse(completion("hello")));
    const res = await model.invoke("hi");
    expect(res.content).toBe("hello");
    expect(res.getType()).toBe("ai");
  });

  it("batch fans out over multiple inputs", async () => {
    const { model } = mockChat(() => jsonResponse(completion("ok")));
    const out = await model.batch(["a", "b", "c"]);
    expect(out).toHaveLength(3);
    expect(out.every((m) => m.content === "ok")).toBe(true);
  });

  it("stream yields message chunks", async () => {
    const { model } = mockChat(() => sseResponse([chunk({ content: "Hel" }), chunk({ content: "lo" }), chunk({}, "stop")]));
    let text = "";
    for await (const c of await model.stream("hi")) text += typeof c.content === "string" ? c.content : "";
    expect(text).toBe("Hello");
  });

  it("composes in an LCEL chain via .pipe()", async () => {
    const { model } = mockChat(() => jsonResponse(completion("Bonjour")));
    const chain = ChatPromptTemplate.fromTemplate("Translate to {lang}: {text}").pipe(model).pipe(new StringOutputParser());
    const out = await chain.invoke({ lang: "French", text: "Hello" });
    expect(out).toBe("Bonjour");
  });

  it("is not lc-serializable and exposes the standard identifiers", () => {
    const model = new ChatInterfaze({ apiKey: "t" });
    expect(model.lc_serializable).toBe(false);
    expect(model._llmType()).toBeTypeOf("string");
    expect(model._modelType()).toBeTypeOf("string");
  });
});
