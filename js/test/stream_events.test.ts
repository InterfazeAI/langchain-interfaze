import { describe, expect, it } from "vitest";
import { chunk, mockChat, sseResponse } from "./helpers.js";

describe(".streamEvents() filtering", () => {
  it("strips side-channel tags from streamed events (v2 protocol)", async () => {
    const chunks = [chunk({ content: "<th" }), chunk({ content: "ink>secret</think>The sky " }), chunk({ content: "is blue." }), chunk({}, "stop")];
    const { model } = mockChat(() => sseResponse(chunks));
    let text = "";
    for await (const ev of model.streamEvents("x", { version: "v2" })) {
      if (ev.event === "on_chat_model_stream") {
        const c = (ev.data as { chunk?: { content?: unknown } }).chunk?.content;
        if (typeof c === "string") text += c;
      }
    }
    expect(text).not.toContain("<think>");
    expect(text).not.toContain("secret");
    expect(text).toBe("The sky is blue.");
  });

  it("strips side-channel tags from streamed events (default content-block protocol)", async () => {
    // Default protocol takes the native fast path the override neutralizes; the v2 case
    // above routes through the Runnable bridge and can't catch a regression here on its own.
    const chunks = [chunk({ content: "<th" }), chunk({ content: "ink>secret</think>The sky " }), chunk({ content: "is blue." }), chunk({}, "stop")];
    const { model } = mockChat(() => sseResponse(chunks));
    let text = "";
    for await (const ev of model.streamEvents("x")) {
      if (ev.event === "content-block-delta" && ev.delta.type === "text-delta") {
        text += ev.delta.text;
      }
    }
    expect(text).not.toContain("<think>");
    expect(text).not.toContain("secret");
    expect(text).toBe("The sky is blue.");
  });
});

// Tool-call streaming takes a different event-synthesis path than tags (tool_call_chunks +
// concat assembly + a separate usage frame), so assert tool calls and usage survive the
// override on both protocols with no __raw_response leaking into any event.
describe(".streamEvents() tool-call + usage streaming", () => {
  // usage-only frame: empty `choices`; `chunk()` always adds one, so build it by hand.
  const usageChunk = {
    id: "req-test",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "interfaze-beta",
    choices: [],
    usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
  };

  function toolCallSse(): unknown[] {
    return [
      chunk({ tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "get_weather", arguments: "" } }] }),
      chunk({ tool_calls: [{ index: 0, function: { arguments: '{"city":' } }] }),
      chunk({ tool_calls: [{ index: 0, function: { arguments: '"Paris"}' } }] }),
      chunk({}, "stop"),
      usageChunk,
    ];
  }

  it("assembles the tool call and usage from the aggregated output, with no __raw_response leak (v2 protocol)", async () => {
    const { model } = mockChat(() => sseResponse(toolCallSse()));
    const rawEvents: unknown[] = [];
    let finalOutput:
      | {
          tool_calls?: Array<{ name?: string; args?: unknown }>;
          usage_metadata?: { input_tokens?: number; output_tokens?: number; total_tokens?: number };
        }
      | undefined;
    for await (const ev of model.streamEvents("x", { version: "v2" })) {
      rawEvents.push(ev);
      if (ev.event === "on_chat_model_end") {
        finalOutput = (ev.data as { output?: typeof finalOutput }).output;
      }
    }

    expect(finalOutput?.tool_calls?.[0]).toMatchObject({ name: "get_weather", args: { city: "Paris" } });
    expect(finalOutput?.usage_metadata).toMatchObject({ input_tokens: 5, output_tokens: 3, total_tokens: 8 });
    expect(JSON.stringify(rawEvents)).not.toContain("__raw_response");
  });

  it("assembles the tool call and usage from content-block events, with no __raw_response leak (default content-block protocol)", async () => {
    const { model } = mockChat(() => sseResponse(toolCallSse()));
    const rawEvents: unknown[] = [];
    let toolCall: { name?: string; args?: unknown } | undefined;
    let usage: { input_tokens?: number; output_tokens?: number; total_tokens?: number } | undefined;
    for await (const ev of model.streamEvents("x")) {
      rawEvents.push(ev);
      if (ev.event === "content-block-finish") {
        const content = ev.content as { type: string; name?: string; args?: unknown };
        if (content.type === "tool_call") toolCall = content;
      }
      if (ev.event === "message-finish") {
        usage = ev.usage;
      }
    }

    expect(toolCall).toMatchObject({ name: "get_weather", args: { city: "Paris" } });
    expect(usage).toMatchObject({ input_tokens: 5, output_tokens: 3, total_tokens: 8 });
    expect(JSON.stringify(rawEvents)).not.toContain("__raw_response");
  });
});
