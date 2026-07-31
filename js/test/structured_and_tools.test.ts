import { describe, expect, it } from "vitest";
import { z } from "zod";
import { completion, jsonResponse, mockChat } from "./helpers.js";

describe("inherited capabilities", () => {
  it("withStructuredOutput parses a JSON response", async () => {
    const body = completion(JSON.stringify({ merchant: "Walmart", total: 144.02 }));
    const { model } = mockChat(() => jsonResponse(body));
    const structured = model.withStructuredOutput(z.object({ merchant: z.string(), total: z.number() }), { method: "jsonMode" });
    const out = (await structured.invoke("extract")) as { merchant: string; total: number };
    expect(out.merchant).toBe("Walmart");
    expect(out.total).toBe(144.02);
  });

  it("bindTools surfaces tool_calls", async () => {
    const toolCall = {
      id: "call_1",
      type: "function",
      function: { name: "get_weather", arguments: JSON.stringify({ city: "Tokyo" }) },
    };
    const body = completion(null, {
      choices: [{ index: 0, message: { role: "assistant", content: null, tool_calls: [toolCall] }, finish_reason: "tool_calls" }],
    });
    const { model } = mockChat(() => jsonResponse(body));
    const bound = model.bindTools([
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Weather for a city",
          parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
        },
      },
    ]);
    const res = await bound.invoke("weather in Tokyo?");
    expect(res.tool_calls?.[0]?.name).toBe("get_weather");
    expect(res.tool_calls?.[0]?.args).toEqual({ city: "Tokyo" });
  });
});
