import { describe, expect, it } from "vitest";
import { HumanMessage } from "@langchain/core/messages";
import { InterfazeError } from "interfaze";
import { completion, jsonResponse, lastBody, mockChat, VIDEO_URL } from "./helpers.js";

function lastContent(calls: ReturnType<typeof mockChat>["calls"]): Array<Record<string, unknown>> {
  const messages = lastBody(calls).messages as Array<{ content: unknown }>;
  return messages.at(-1)!.content as Array<Record<string, unknown>>;
}

describe("video content blocks", () => {
  it("rewrites a url video block to a file part and infers its mime type", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion()));
    await model.invoke([
      new HumanMessage({
        content: [
          { type: "text", text: "what happens?" },
          { type: "video", url: VIDEO_URL },
        ] as never,
      }),
    ]);
    expect(lastContent(calls)).toContainEqual({ type: "file", file: { file_data: VIDEO_URL, format: "video/mp4" } });
  });

  it("omits format when the url has no recognizable extension", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion()));
    await model.invoke([new HumanMessage({ content: [{ type: "video", url: "https://example.com/clip" }] as never })]);
    expect(lastContent(calls)[0]!.file).toEqual({ file_data: "https://example.com/clip" });
  });

  it("rewrites a base64 video block with mime type", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion()));
    await model.invoke([new HumanMessage({ content: [{ type: "video", base64: "AAAA", mime_type: "video/mp4" }] as never })]);
    const part = lastContent(calls)[0]!;
    expect(part).toEqual({ type: "file", file: { file_data: "data:video/mp4;base64,AAAA", format: "video/mp4" } });
  });

  it("rejects a file_id video block (interfaze has no file store)", async () => {
    const { model } = mockChat(() => jsonResponse(completion()));
    await expect(model.invoke([new HumanMessage({ content: [{ type: "video", file_id: "file-123" }] as never })])).rejects.toThrow(/file_id/);
  });

  it("forwards extras.filename", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion()));
    await model.invoke([new HumanMessage({ content: [{ type: "video", url: VIDEO_URL, extras: { filename: "clip.mp4" } }] as never })]);
    const file = lastContent(calls)[0]!.file as Record<string, unknown>;
    expect(file).toEqual({ file_data: VIDEO_URL, format: "video/mp4", filename: "clip.mp4" });
  });

  // url: null is the natural shape from a deserialized message
  it("falls through an explicit null url to base64", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion()));
    await model.invoke([new HumanMessage({ content: [{ type: "video", url: null, base64: "AAAA" }] as never })]);
    expect(lastContent(calls)[0]!.file).toEqual({ file_data: "data:video/mp4;base64,AAAA", format: "video/mp4" });
  });

  it("ignores an explicit null file_id", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion()));
    await model.invoke([new HumanMessage({ content: [{ type: "video", url: VIDEO_URL, file_id: null }] as never })]);
    expect(lastContent(calls)[0]!.file).toEqual({ file_data: VIDEO_URL, format: "video/mp4" });
  });

  it("rejects a file_id on a non-video block too (no file store)", async () => {
    const { model } = mockChat(() => jsonResponse(completion()));
    await expect(model.invoke([new HumanMessage({ content: [{ type: "file", file_id: "file-123" }] as never })])).rejects.toThrow(/file_id/);
  });

  it("throws when a video block has no source", async () => {
    const { model } = mockChat(() => jsonResponse(completion()));
    await expect(model.invoke([new HumanMessage({ content: [{ type: "video" }] as never })])).rejects.toThrow(InterfazeError);
  });
});
