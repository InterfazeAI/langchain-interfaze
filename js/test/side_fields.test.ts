import { describe, expect, it } from "vitest";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { ChatInterfaze } from "../src/index.js";
import { chunk, completion, envelopeChunk, jsonResponse, mockChat, sseResponse } from "./helpers.js";

const PC = [{ name: "ocr", result: { extracted_text: "x" } }];

describe("non-streaming side fields", () => {
  it("surfaces precontext/reasoning/vcache on both metadata maps", async () => {
    const { model } = mockChat(() => jsonResponse(completion("Hello there", { precontext: PC, reasoning: "because reasons", vcache: true })));
    const res = (await model.invoke("hi")) as AIMessage;
    expect(res.response_metadata.precontext).toEqual(PC);
    expect(res.response_metadata.reasoning).toBe("because reasons");
    expect(res.response_metadata.vcache).toBe(true);
    expect(res.additional_kwargs.precontext).toEqual(PC);
    expect(res.additional_kwargs.reasoning).toBe("because reasons");
    expect(res.additional_kwargs.vcache).toBe(true);
  });

  it("leaves plain responses untouched and never leaks __raw_response", async () => {
    const { model } = mockChat(() => jsonResponse(completion("Hi!")));
    const res = (await model.invoke("hi")) as AIMessage;
    expect(res.content).toBe("Hi!");
    expect("precontext" in res.response_metadata).toBe(false);
    expect("reasoning" in res.response_metadata).toBe(false);
    expect(res.response_metadata.vcache).toBe(false);
    expect("__raw_response" in res.additional_kwargs).toBe(false);
  });

  it("keeps generation.text in step with the stripped content", async () => {
    const { model } = mockChat(() => jsonResponse(completion("<think>SECRET</think>The answer is 42")));
    const res = await model.generate([[new HumanMessage("x")]]);
    expect(res.generations[0]![0]!.text).toBe("The answer is 42");
  });

  it("does not let an empty envelope value block the inline payload", async () => {
    const content = '<precontext>[{"name":"ocr"}]</precontext>The sky is blue.';
    const { model } = mockChat(() => jsonResponse(completion(content, { precontext: [] })));
    const res = (await model.invoke("x")) as AIMessage;
    expect(res.response_metadata.precontext).toEqual([{ name: "ocr" }]);
  });

  it("keeps a truncated <precontext> out of invoke() content", async () => {
    const { model } = mockChat(() =>
      jsonResponse(
        completion('Total is <precontext>[{"ssn":"123-45-6789"', {
          choices: [{ index: 0, message: { role: "assistant", content: 'Total is <precontext>[{"ssn":"123-45-6789"' }, finish_reason: "length" }],
        })
      )
    );
    const res = (await model.invoke("x")) as AIMessage;
    expect(String(res.content)).not.toContain("123-45-6789");
  });

  it("keeps prose that mentions a tag intact", async () => {
    const { model } = mockChat(() => jsonResponse(completion("Wrap metadata in <precontext> tags, then continue.")));
    const res = (await model.invoke("x")) as AIMessage;
    expect(res.content).toBe("Wrap metadata in <precontext> tags, then continue.");
  });

  it("strips inline <think>/<precontext> tags from content", async () => {
    const content = "<think>Rayleigh scattering.</think>" + '<precontext>[{"name":"ocr","result":{"x":1}}]</precontext>' + "The sky is blue.";
    const { model } = mockChat(() => jsonResponse(completion(content)));
    const res = (await model.invoke("why is the sky blue?")) as AIMessage;
    expect(res.content).toBe("The sky is blue.");
    expect(res.response_metadata.reasoning).toBe("Rayleigh scattering.");
    expect(res.response_metadata.precontext).toEqual([{ name: "ocr", result: { x: 1 } }]);
  });
});

describe("identifying params", () => {
  it("fingerprints the api key and header values instead of publishing them", () => {
    const { model } = mockChat(() => jsonResponse(completion()));
    const params = model._identifyingParams() as unknown as Record<string, unknown>;
    expect(JSON.stringify(params)).not.toContain("sk-test");
    expect("apiKey" in params).toBe(false);
    expect("defaultHeaders" in params).toBe(false);
  });

  it("keeps two header values in separate cache entries", () => {
    const a = new ChatInterfaze({ apiKey: "k", configuration: { defaultHeaders: { "x-tenant": "a" } } });
    const b = new ChatInterfaze({ apiKey: "k", configuration: { defaultHeaders: { "x-tenant": "b" } } });
    expect(JSON.stringify(a._identifyingParams())).not.toEqual(JSON.stringify(b._identifyingParams()));
  });

  it("shows the flags it owns in the clear", () => {
    const model = new ChatInterfaze({ apiKey: "k", bypassCache: true });
    expect((model._identifyingParams() as unknown as Record<string, unknown>).interfazeHeaders).toEqual(["x-interfaze-bypass-cache=true"]);
  });
});

describe("stream ordering", () => {
  it("attaches envelope side fields to the chunk they arrived with", async () => {
    const { model } = mockChat(() =>
      sseResponse([chunk({ content: "a" }), envelopeChunk({ vcache: true }), chunk({ content: "b" }), chunk({}, "stop")])
    );
    const seen: Array<[string, unknown]> = [];
    for await (const c of await model.stream("x")) seen.push([String(c.content), c.additional_kwargs.vcache]);
    // not buffered to the end: vcache lands before the last content chunk
    const at = seen.findIndex(([, v]) => v === true);
    expect(at).toBeGreaterThanOrEqual(0);
    expect(seen.slice(at).some(([text]) => text === "b")).toBe(true);
  });

  it("still delivers side fields when the consumer breaks early", async () => {
    const { model } = mockChat(() =>
      sseResponse([chunk({ content: "a" }), envelopeChunk({ vcache: true }), chunk({ content: "b" }), chunk({}, "stop")])
    );
    let vcache: unknown;
    for await (const c of await model.stream("x")) {
      vcache ??= c.additional_kwargs.vcache;
      if (c.content === "b") break;
    }
    expect(vcache).toBe(true);
  });
});
