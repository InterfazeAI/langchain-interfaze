import { describe, expect, it } from "vitest";
import { chunk, lastBody, mockChat, sseResponse } from "./helpers.js";

async function collect(model: { stream: (i: string) => Promise<AsyncIterable<{ content: unknown; additional_kwargs: Record<string, unknown> }>> }) {
  const out: Array<{ content: unknown; additional_kwargs: Record<string, unknown> }> = [];
  for await (const c of await model.stream("x")) out.push(c);
  return out;
}

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

  // Interfaze only reports usage on a stream when asked; keep this in step with the
  // python package, where langchain-openai leaves it off for non-OpenAI base URLs.
  it("asks the server for streamed usage", async () => {
    const chunks = [chunk({ content: "hi" }), chunk({}, "stop")];
    const { model, calls } = mockChat(() => sseResponse(chunks));
    for await (const _ of await model.stream("x")) void _;
    expect(lastBody(calls).stream_options).toEqual({ include_usage: true });
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
