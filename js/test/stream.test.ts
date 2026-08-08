import { describe, expect, it } from "vitest";
import { concat } from "@langchain/core/utils/stream";
import { isAIMessage } from "@langchain/core/messages";
import { chunk, completion, envelopeChunk, jsonResponse, lastBody, mockChat, sseResponse } from "./helpers.js";

async function concatAll(model: { stream: (i: string) => Promise<AsyncIterable<any>> }) {
  let merged: any;
  for await (const c of await model.stream("x")) merged = merged === undefined ? c : concat(merged, c);
  return merged;
}

async function collect(model: { stream: (i: string) => Promise<AsyncIterable<{ content: unknown; additional_kwargs: Record<string, unknown> }>> }) {
  const out: Array<{ content: unknown; additional_kwargs: Record<string, unknown> }> = [];
  for await (const c of await model.stream("x")) out.push(c);
  return out;
}

// Interfaze sends `role` only on the first delta. If it ever sends none at all,
// @langchain/openai yields ChatMessageChunk rather than AIMessageChunk — an
// `instanceof AIMessageChunk` gate here would skip the filter and leak raw tags.
describe("role-less deltas", () => {
  const roleless = (content: string, finish: string | null = null) => ({
    id: "req-test",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "interfaze-beta",
    choices: [{ index: 0, delta: content ? { content } : {}, finish_reason: finish }],
  });

  it("still strips tags and stamps model_provider", async () => {
    const frames = [roleless("<th"), roleless("ink>secret</think>The sky "), roleless("is blue."), roleless("", "stop")];
    const { model } = mockChat(() => sseResponse(frames as never));
    let text = "";
    const providers: unknown[] = [];
    for await (const c of await model.stream("x")) {
      text += typeof c.content === "string" ? c.content : "";
      providers.push(c.response_metadata.model_provider);
    }
    expect(text).toBe("The sky is blue.");
    expect(text).not.toContain("<think>");
    expect(new Set(providers)).toEqual(new Set(["interfaze"]));
  });

  // The role is normalized at the converter, so these stay AIMessageChunk instead of
  // degrading to the generic ChatMessageChunk (which carries no additional_kwargs).
  it("yields AIMessageChunk and keeps envelope side fields", async () => {
    const frames = [{ ...roleless("Hello "), precontext: [{ name: "ocr" }], vcache: true }, roleless("world"), roleless("", "stop")];
    const { model } = mockChat(() => sseResponse(frames as never));
    const got = await collect(model as never);
    const merged = await (async () => {
      let m: any;
      for (const c of got) m = m === undefined ? c : concat(m, c);
      return m;
    })();
    expect(new Set(got.map((c) => c.constructor.name))).toEqual(new Set(["AIMessageChunk"]));
    expect(isAIMessage(merged)).toBe(true);
    expect(merged.additional_kwargs.precontext).toEqual([{ name: "ocr" }]);
    expect(merged.additional_kwargs.vcache).toBe(true);
  });
});

describe("streaming side-channel filter", () => {
  it("strips inline precontext and carries it on a chunk", async () => {
    const chunks = [
      chunk({ content: '<precontext>[{"name":"ocr","result":{"extracted_text":"x"}}]</precontext>' }),
      chunk({ content: "Total " }),
      chunk({ content: "is $12.34" }),
      chunk({}, "stop"),
    ];
    const { model } = mockChat(() => sseResponse(chunks));
    const got = await collect(model as never);
    const text = got.map((c) => (typeof c.content === "string" ? c.content : "")).join("");
    expect(text).not.toContain("<precontext>");
    expect(text).toBe("Total is $12.34");
    const withPc = got.filter((c) => c.additional_kwargs.precontext);
    expect(withPc.length).toBeGreaterThan(0);
    expect((withPc[0]!.additional_kwargs.precontext as Array<{ name: string }>)[0]!.name).toBe("ocr");
  });

  it("recovers <think> reasoning split across chunk boundaries", async () => {
    const chunks = [
      chunk({ content: "<th" }),
      chunk({ content: "ink>Rayleigh scat" }),
      chunk({ content: "tering.</think>The sky " }),
      chunk({ content: "is blue." }),
      chunk({}, "stop"),
    ];
    const { model } = mockChat(() => sseResponse(chunks));
    const got = await collect(model as never);
    const text = got.map((c) => (typeof c.content === "string" ? c.content : "")).join("");
    expect(text).not.toContain("<think>");
    expect(text).toBe("The sky is blue.");
    const reasoning = got.filter((c) => c.additional_kwargs.reasoning);
    expect(reasoning[0]!.additional_kwargs.reasoning).toBe("Rayleigh scattering.");
  });

  // langchain-openai leaves this off for non-OpenAI base URLs, so we opt in.
  it("asks the server for streamed usage", async () => {
    const chunks = [chunk({ content: "hi" }), chunk({}, "stop")];
    const { model, calls } = mockChat(() => sseResponse(chunks));
    for await (const _ of await model.stream("x")) void _;
    expect(lastBody(calls).stream_options).toEqual({ include_usage: true });
  });

  it("applies a repeated envelope side field only once", async () => {
    const pc = [{ name: "ocr" }];
    const chunks = [chunk({ content: "a" }, null, { precontext: pc }), chunk({ content: "b" }, null, { precontext: pc }), chunk({}, "stop")];
    const { model } = mockChat(() => sseResponse(chunks));
    const got = await collect(model as never);
    expect(got.filter((c) => c.additional_kwargs.precontext)).toHaveLength(1);
  });

  it("keeps distinct envelope side fields from every chunk", async () => {
    const chunks = [
      chunk({ content: "a" }, null, { precontext: [{ name: "ocr" }] }),
      chunk({ content: "b" }, null, { precontext: [{ name: "web_search" }] }),
      chunk({}, "stop"),
    ];
    const { model } = mockChat(() => sseResponse(chunks));
    const got = await collect(model as never);
    const names = got.flatMap((c) => ((c.additional_kwargs.precontext as Array<{ name: string }>) ?? []).map((p) => p.name));
    expect(names).toEqual(["ocr", "web_search"]);
  });

  it("surfaces side fields riding a choice-less usage frame", async () => {
    const chunks = [
      chunk({ content: "hi" }),
      chunk({}, "stop"),
      envelopeChunk({
        usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        precontext: [{ name: "ocr" }],
        reasoning: "wire",
        vcache: true,
      }),
    ];
    const { model } = mockChat(() => sseResponse(chunks));
    const merged = await concatAll(model as never);
    expect(merged.additional_kwargs.precontext).toEqual([{ name: "ocr" }]);
    expect(merged.additional_kwargs.reasoning).toBe("wire");
    expect(merged.additional_kwargs.vcache).toBe(true);
    // Observing the frame rather than reshaping it: a fabricated choice would make the
    // parent stamp usage twice, and _mergeDicts sums numbers.
    expect(merged.response_metadata.usage).toEqual({ prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 });
  });

  it("delivers an envelope that arrives before the first assistant delta", async () => {
    const chunks = [envelopeChunk({ precontext: [{ name: "ocr" }], vcache: true }), chunk({ content: "hi" }), chunk({}, "stop")];
    const { model } = mockChat(() => sseResponse(chunks));
    const merged = await concatAll(model as never);
    expect(merged.additional_kwargs.precontext).toEqual([{ name: "ocr" }]);
    expect(merged.additional_kwargs.vcache).toBe(true);
  });

  it("adds no extra chunk to a plain stream", async () => {
    const { model } = mockChat(() => sseResponse([chunk({ content: "hi" }), chunk({}, "stop")]));
    expect(await collect(model as never)).toHaveLength(2);
  });

  // A truncated response leaves the tag open; the buffered text must not vanish.
  it("recovers text from an unterminated tag", async () => {
    const chunks = [chunk({ content: "<think>never closed and the real answer 42" }), chunk({}, "length")];
    const { model } = mockChat(() => sseResponse(chunks));
    const got = await collect(model as never);
    const text = got.map((c) => (typeof c.content === "string" ? c.content : "")).join("");
    // Matches the SDK: an unmatched tag survives verbatim rather than being swallowed.
    expect(text).toBe("<think>never closed and the real answer 42");
  });

  // Non-streaming has the whole body, so an unmatched tag is prose and must survive
  // verbatim — stripping it would mangle any answer that mentions the tag name.
  it("leaves an unmatched tag alone when not streaming", async () => {
    const { model } = mockChat(() => jsonResponse(completion("Wrap your reasoning in <think> tags.")));
    expect((await model.invoke("x")).content).toBe("Wrap your reasoning in <think> tags.");
  });

  it("does not duplicate the prefix when the tag opens mid-text", async () => {
    const chunks = [chunk({ content: "The answer is 42. <think>because reasons" }), chunk({}, "length")];
    const { model } = mockChat(() => sseResponse(chunks));
    const got = await collect(model as never);
    const text = got.map((c) => (typeof c.content === "string" ? c.content : "")).join("");
    expect(text).toBe("The answer is 42. <think>because reasons");
  });

  it("keeps an envelope side field alongside the inline one", async () => {
    const chunks = [
      chunk({ content: "<think>INLINE</think>Hi" }),
      chunk({}, "stop"),
      envelopeChunk({ reasoning: "ENVELOPE", usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } }),
    ];
    const { model } = mockChat(() => sseResponse(chunks));
    const merged = await concatAll(model as never);
    expect(String(merged.additional_kwargs.reasoning)).toContain("ENVELOPE");
    expect(String(merged.additional_kwargs.reasoning)).toContain("INLINE");
  });

  it("keeps every distinct choice-less envelope frame", async () => {
    const chunks = [
      envelopeChunk({ precontext: [{ name: "ocr" }] }),
      chunk({ content: "hi" }),
      envelopeChunk({ precontext: [{ name: "web_search" }] }),
      chunk({}, "stop"),
    ];
    const { model } = mockChat(() => sseResponse(chunks));
    const merged = await concatAll(model as never);
    expect(((merged.additional_kwargs.precontext as Array<{ name: string }>) ?? []).map((p) => p.name)).toEqual(["ocr", "web_search"]);
  });

  it("surfaces an empty precontext array rather than dropping the key", async () => {
    const chunks = [chunk({ content: "hi" }, "stop"), envelopeChunk({ precontext: [], vcache: false })];
    const { model } = mockChat(() => sseResponse(chunks));
    const merged = await concatAll(model as never);
    expect(merged.additional_kwargs).toHaveProperty("precontext");
  });

  it("emits no side-channel chunk for plain content", async () => {
    const chunks = [chunk({ content: "Hello " }), chunk({ content: "world" }), chunk({}, "stop")];
    const { model } = mockChat(() => sseResponse(chunks));
    const got = await collect(model as never);
    const text = got.map((c) => (typeof c.content === "string" ? c.content : "")).join("");
    expect(text).toBe("Hello world");
    expect(got.some((c) => c.additional_kwargs.precontext || c.additional_kwargs.reasoning)).toBe(false);
    expect(got.some((c) => "__raw_response" in c.additional_kwargs)).toBe(false);
  });
});
