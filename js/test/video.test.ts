import { describe, expect, it } from "vitest";
import { HumanMessage } from "@langchain/core/messages";
import { InterfazeError } from "interfaze";
import { completion, jsonResponse, lastBody, mockChat, VIDEO_URL } from "./helpers.js";

function lastContent(calls: ReturnType<typeof mockChat>["calls"]): Array<Record<string, unknown>> {
  const messages = lastBody(calls).messages as Array<{ content: unknown }>;
  return messages.at(-1)!.content as Array<Record<string, unknown>>;
}

describe("video content blocks", () => {
  it("rewrites a url video block to a file part", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion()));
    await model.invoke([
      new HumanMessage({
        content: [
          { type: "text", text: "what happens?" },
          { type: "video", url: VIDEO_URL },
        ] as never,
      }),
    ]);
    expect(lastContent(calls)).toContainEqual({ type: "file", file: { file_data: VIDEO_URL } });
  });

  it("rewrites a base64 video block with mime type", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion()));
    await model.invoke([new HumanMessage({ content: [{ type: "video", base64: "AAAA", mime_type: "video/mp4" }] as never })]);
    const part = lastContent(calls)[0]!;
    expect(part).toEqual({ type: "file", file: { file_data: "data:video/mp4;base64,AAAA", format: "video/mp4" } });
  });

  it("rewrites a file_id video block", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion()));
    await model.invoke([new HumanMessage({ content: [{ type: "video", file_id: "file-123" }] as never })]);
    expect(lastContent(calls)[0]).toEqual({ type: "file", file: { file_id: "file-123" } });
  });

  it("forwards extras.filename", async () => {
    const { model, calls } = mockChat(() => jsonResponse(completion()));
    await model.invoke([new HumanMessage({ content: [{ type: "video", url: VIDEO_URL, extras: { filename: "clip.mp4" } }] as never })]);
    const file = lastContent(calls)[0]!.file as Record<string, unknown>;
    expect(file).toEqual({ file_data: VIDEO_URL, filename: "clip.mp4" });
  });

  it("throws when a video block has no source", async () => {
    const { model } = mockChat(() => jsonResponse(completion()));
    await expect(model.invoke([new HumanMessage({ content: [{ type: "video" }] as never })])).rejects.toThrow(InterfazeError);
  });
});
