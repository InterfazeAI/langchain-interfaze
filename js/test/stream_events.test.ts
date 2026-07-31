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
    // No `version` option: this is the newer content-block-centric protocol, which
    // ChatOpenAICompletions serves via a *native* `_streamChatModelEvents` fast path
    // that bypasses `_streamResponseChunks` (and therefore our side-channel filter)
    // unless neutralized. The v2 case above routes through the legacy Runnable
    // bridge and never touches that fast path, so it can't catch this on its own.
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

/**
 * `_streamChatModelEvents` is a thin override that just re-points both `.streamEvents()`
 * protocols at our filtered `_streamResponseChunks` (see the comment on that override in
 * src/chat_models.ts). The tests above only exercise the side-channel-tag path through it.
 * Tool-call streaming is the other consumer of `_streamResponseChunks`'s per-chunk
 * `additional_kwargs.__raw_response` bookkeeping (applySideFields + delete), and it takes a
 * materially different path through `@langchain/core`'s event synthesis (tool_call_chunks +
 * concat-based assembly, plus the separate usage-only chunk), so it needs its own coverage:
 * assembled tool calls and usage must survive the override on both protocols, with no
 * `__raw_response` leaking into any emitted event.
 */
describe(".streamEvents() tool-call + usage streaming", () => {
  // Trailing chunk shaped like a real OpenAI usage-only frame: empty `choices`, populated
  // `usage`. `chunk()` always builds a non-empty `choices` array, so this is constructed by
  // hand instead.
  const usageChunk = {
    id: "req-test",
    object: "chat.completion.chunk",
    created: 1_700_000_000,
    model: "interfaze-beta",
    choices: [],
    usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
  };

  // Three deltas building one tool call (`get_weather({ city: "Paris" })`) across the id/name
  // frame and two incremental-argument frames, a `stop` frame, then the usage frame.
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

    // (a) the assembled tool call surfaces on the final aggregated message.
    expect(finalOutput?.tool_calls?.[0]).toMatchObject({ name: "get_weather", args: { city: "Paris" } });
    // (b) usage_metadata is present on the aggregated output.
    expect(finalOutput?.usage_metadata).toMatchObject({ input_tokens: 5, output_tokens: 3, total_tokens: 8 });
    // (c) no emitted event carries a leaked __raw_response anywhere in its payload.
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

    // (a) the assembled tool call surfaces once its content block finishes.
    expect(toolCall).toMatchObject({ name: "get_weather", args: { city: "Paris" } });
    // (b) usage is present on the message-finish event.
    expect(usage).toMatchObject({ input_tokens: 5, output_tokens: 3, total_tokens: 8 });
    // (c) no emitted event carries a leaked __raw_response anywhere in its payload.
    expect(JSON.stringify(rawEvents)).not.toContain("__raw_response");
  });
});
