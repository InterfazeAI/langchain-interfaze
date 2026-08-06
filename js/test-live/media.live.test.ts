import { describe, expect, it } from "vitest";
import { z } from "zod";
import { HumanMessage } from "@langchain/core/messages";
import { chat, filePart, FILES, hasKey, lower, precontextNames, SLOW, text, videoPart } from "./helpers.js";

const llm = chat();

function ask(prompt: string, part: Record<string, unknown>): HumanMessage {
  return new HumanMessage({ content: [{ type: "text", text: prompt }, part] as never });
}

describe.skipIf(!hasKey)("audio", () => {
  it("stt_basic", { timeout: SLOW }, async () => {
    const out = await llm
      .withStructuredOutput(z.object({ text: z.string() }), { includeRaw: true })
      .invoke([ask("Transcribe the audio file", filePart(FILES.sttShort, "stt_medical_short.mp4"))]);
    expect(lower((out.parsed as { text: string }).text)).toContain("amoxicillin");
    expect(precontextNames(out.raw as never).join(" ")).toMatch(/stt|speech_to_text/);
  });

  it("stt_diarization", { timeout: SLOW }, async () => {
    const schema = z.object({
      full_text: z.string(),
      chunks: z.array(z.object({ speaker_id: z.string(), text: z.string(), start_time: z.number(), end_time: z.number() })),
      number_of_speakers: z.number().int(),
    });
    const out = await llm
      .withStructuredOutput(schema, { includeRaw: true })
      .invoke([ask("Transcribe and identify the speakers in the audio file", filePart(FILES.sttMulti, "stt_multispeaker.mp3"))]);
    const parsed = out.parsed as z.infer<typeof schema>;
    expect(parsed.number_of_speakers).toBeGreaterThanOrEqual(2);
    expect(parsed.chunks.length).toBeGreaterThan(5);
    expect(precontextNames(out.raw as never).length).toBeGreaterThan(0);
  });

  it("stt_translate", { timeout: SLOW }, async () => {
    const schema = z.object({
      translated_text: z.string(),
      original_language_code: z.string(),
      translated_language_code: z.string(),
    });
    const out = await llm.withStructuredOutput(schema).invoke(`Transcribe the audio file and translate it to chinese ${FILES.sttShort}`);
    expect(["zh", "zh-cn", "zh-tw"]).toContain(lower(out.translated_language_code));
  });

  it("stt_summary", { timeout: SLOW }, async () => {
    const out = await llm
      .withStructuredOutput(z.object({ text: z.string(), summary: z.string(), intent: z.string() }))
      .invoke(`Transcribe the audio file and summarize it ${FILES.sttCall}`);
    expect(out.text.length).toBeGreaterThan(0);
    expect(out.summary.length).toBeGreaterThan(0);
    expect(out.intent.length).toBeGreaterThan(0);
  });
});

describe.skipIf(!hasKey)("video", () => {
  it("video_describe via the {type:'video'} content block", { timeout: SLOW }, async () => {
    const res = await llm.invoke([ask("Describe what happens in this video in one or two sentences.", videoPart(FILES.video))]);
    const body = lower(text(res));
    expect(body.length).toBeGreaterThan(0);
    expect(body).toMatch(/rabbit|bunny|forest|tree|grass|animal|burrow|meadow|field|nature/);
  });
});
