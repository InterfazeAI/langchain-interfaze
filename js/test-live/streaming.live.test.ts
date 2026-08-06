import { describe, expect, it } from "vitest";
import { HumanMessage } from "@langchain/core/messages";
import { chat, FAST, freshChat, hasKey, IMAGES, imagePart, lower, SLOW } from "./helpers.js";

const llm = chat();

async function collect(stream: AsyncIterable<{ content: unknown }>): Promise<string> {
  let out = "";
  for await (const c of stream) out += typeof c.content === "string" ? c.content : "";
  return out;
}

describe.skipIf(!hasKey)("streaming", () => {
  it("stream_haiku", { timeout: FAST }, async () => {
    const out = await collect(await llm.stream("Write a haiku about coding"));
    expect(out.length).toBeGreaterThan(10);
    expect(out).not.toContain("<think>");
    expect(out).not.toContain("<precontext>");
  });

  it("stream_capital", { timeout: FAST }, async () => {
    const out = await collect(await llm.stream("What is the capital of France? Answer in one word."));
    expect(lower(out)).toContain("paris");
  });

  it("stream_reasoning keeps <think> out of the visible text", { timeout: FAST }, async () => {
    const fresh = freshChat();
    const chunks = [];
    for await (const c of await fresh.stream("Write a haiku about streaming data", { reasoningEffort: "high" })) {
      chunks.push(c);
    }
    const visible = chunks.map((c) => (typeof c.content === "string" ? c.content : "")).join("");
    expect(visible).not.toContain("<think>");
    expect(visible).not.toContain("</think>");
    const reasoning = chunks.map((c) => c.additional_kwargs?.reasoning).filter(Boolean);
    expect(reasoning.length).toBeGreaterThan(0);
  });

  it("token callbacks never see the raw side channels", { timeout: FAST }, async () => {
    const fresh = freshChat();
    const tokens: string[] = [];
    await collect(
      await fresh.stream("Write a haiku about streaming data", {
        reasoningEffort: "high",
        callbacks: [{ handleLLMNewToken: (t: string) => void tokens.push(t) }],
      })
    );
    expect(tokens.join("")).not.toContain("<think>");
  });

  it("streamEvents routes through the filtered chunk path", { timeout: FAST }, async () => {
    const fresh = freshChat();
    let evText = "";
    for await (const ev of fresh.streamEvents("Write a haiku about streaming data", { version: "v2", reasoningEffort: "high" })) {
      if (ev.event === "on_chat_model_stream") {
        const content = (ev.data as { chunk?: { content?: unknown } }).chunk?.content;
        if (typeof content === "string") evText += content;
      }
    }
    expect(evText.length).toBeGreaterThan(0);
    expect(evText).not.toContain("<think>");
  });

  it("streams inline precontext when showAdditionalInfo is on", { timeout: SLOW }, async () => {
    const verbose = chat({ showAdditionalInfo: true, bypassCache: true });
    const chunks = [];
    for await (const c of await verbose.stream([
      new HumanMessage({
        content: [{ type: "text", text: "Where is this store located?" }, imagePart(IMAGES.receipt)] as never,
      }),
    ])) {
      chunks.push(c);
    }
    const visible = chunks.map((c) => (typeof c.content === "string" ? c.content : "")).join("");
    expect(visible).not.toContain("<precontext>");
    const precontext = chunks.flatMap((c) => (c.additional_kwargs?.precontext as unknown[]) ?? []);
    expect(precontext.length).toBeGreaterThan(0);
  });
});
