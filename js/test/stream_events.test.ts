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
