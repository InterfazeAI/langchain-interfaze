import { describe, expect, it } from "vitest";
import { SideChannelFilter, stripSideChannels } from "../src/side_channels.js";

describe("stripSideChannels", () => {
  it("pulls out reasoning and precontext, returns clean text", () => {
    const out = stripSideChannels('<think>because</think><precontext>[{"name":"ocr"}]</precontext>The sky is blue.');
    expect(out.text).toBe("The sky is blue.");
    expect(out.reasoning).toBe("because");
    expect(out.precontext).toEqual([{ name: "ocr" }]);
  });

  it("leaves plain text untouched", () => {
    const out = stripSideChannels("just text");
    expect(out.text).toBe("just text");
    expect(out.reasoning).toBeUndefined();
    expect(out.precontext).toBeUndefined();
  });

  it("ignores a malformed precontext block", () => {
    const out = stripSideChannels("<precontext>not json</precontext>hi");
    expect(out.text).toBe("hi");
    expect(out.precontext).toBeUndefined();
  });
});

describe("SideChannelFilter", () => {
  it("strips a <think> block split across feed() calls", () => {
    const f = new SideChannelFilter();
    let out = "";
    for (const piece of ["<th", "ink>secret</think>The sky ", "is blue."]) out += f.feed(piece);
    out += f.flush();
    expect(out).toBe("The sky is blue.");
  });

  it("passes plain content straight through", () => {
    const f = new SideChannelFilter();
    expect(f.feed("Hello ") + f.feed("world") + f.flush()).toBe("Hello world");
  });

  it("drops an unterminated side-channel block on flush", () => {
    const f = new SideChannelFilter();
    expect(f.feed("<think>partial")).toBe("");
    expect(f.flush()).toBe("");
  });
});
