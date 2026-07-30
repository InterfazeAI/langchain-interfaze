import { describe, expect, it } from "vitest";
import { AIMessage } from "@langchain/core/messages";
import { completion, jsonResponse, mockChat } from "./helpers.js";

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

  it("strips inline <think>/<precontext> tags from content", async () => {
    const content = "<think>Rayleigh scattering.</think>" + '<precontext>[{"name":"ocr","result":{"x":1}}]</precontext>' + "The sky is blue.";
    const { model } = mockChat(() => jsonResponse(completion(content)));
    const res = (await model.invoke("why is the sky blue?")) as AIMessage;
    expect(res.content).toBe("The sky is blue.");
    expect(res.response_metadata.reasoning).toBe("Rayleigh scattering.");
    expect(res.response_metadata.precontext).toEqual([{ name: "ocr", result: { x: 1 } }]);
  });
});
