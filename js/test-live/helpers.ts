import { readFileSync } from "node:fs";
import { ChatInterfaze, type ChatInterfazeFields } from "../src/index.js";

export const hasKey = !!process.env.INTERFAZE_API_KEY;
export const BASE_URL = process.env.INTERFAZE_BASE_URL;

/** Live calls that run internal tools (OCR / STT / scrape / forecast) are slow. */
export const SLOW = 300_000;
export const FAST = 120_000;

export function chat(fields: Partial<ChatInterfazeFields> = {}): ChatInterfaze {
  return new ChatInterfaze({
    maxRetries: 1,
    ...fields,
    ...(BASE_URL ? { configuration: { baseURL: BASE_URL, ...fields.configuration } } : {}),
  });
}

/** Fresh model output — the semantic cache otherwise replays a prior answer. */
export function freshChat(fields: Partial<ChatInterfazeFields> = {}): ChatInterfaze {
  return chat({ bypassCache: true, ...fields });
}

export const IMAGES = {
  receipt: "https://jigsawstack.com/preview/vocr-example.jpg",
  receiptItems:
    "https://cdn.hashnode.com/res/hashnode/image/upload/v1741819852493/10b20478-03da-4ed9-be86-0dc33e97a673.jpeg?auto=compress,format&format=webp",
  idMedium: "https://miro.medium.com/v2/resize:fit:698/1*q_FimDPBNMvJXJyDtXT3Jg.jpeg",
  idJpg: "https://r2public.jigsawstack.com/interfaze/examples/id.jpg",
  multilang:
    "https://cdn.hashnode.com/res/hashnode/image/upload/v1746576859594/31e54f33-e825-4930-8fe3-8a1380ba9e16.jpeg?auto=compress,format&format=webp",
  katana: "https://jigsawstack.com/preview/object-detection-example-input.jpg",
  bus: "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
  guiForm: "https://r2public.jigsawstack.com/interfaze/examples/GUI_form.png",
  construction: "https://r2public.jigsawstack.com/interfaze/examples/construction.png",
  gore: "https://plus.unsplash.com/premium_photo-1695691596554-1a07f2a8cf34?q=80&w=1587&auto=format&fit=crop",
  missing: "https://jigsawstack.com/preview/this-image-definitely-does-not-exist-xyz123.jpg",
} as const;

export const FILES = {
  attentionPdf: "https://arxiv.org/pdf/1706.03762",
  sttShort: "https://r2public.jigsawstack.com/interfaze/examples/stt_medical_short.mp4",
  sttMulti: "https://r2public.jigsawstack.com/interfaze/examples/stt_multispeaker.mp3",
  sttCall: "https://r2public.jigsawstack.com/interfaze/examples/stt_call.mp3",
  video: "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
} as const;

const FIXTURES = process.env.INTERFAZE_FIXTURES ?? `${process.env.HOME}/interfaze-sdk-tests/fixtures`;

let receiptCache: string | undefined;
/** base64 JPEG receipt — GT: "The Marco Polo Kitch" / 15.15 / 2018-05-06. */
export function receiptB64(): string {
  receiptCache ??= readFileSync(`${FIXTURES}/receipt.b64`, "utf8").trim();
  return receiptCache;
}

export function imagePart(url: string): Record<string, unknown> {
  return { type: "image_url", image_url: { url } };
}

export function filePart(url: string, filename?: string): Record<string, unknown> {
  return { type: "file", file: { file_data: url, ...(filename ? { filename } : {}) } };
}

export function audioPart(url: string, format = "mp3"): Record<string, unknown> {
  return { type: "input_audio", input_audio: { data: url, format } };
}

export function videoPart(url: string): Record<string, unknown> {
  return { type: "video", url };
}

/** Names of the internal tools Interfaze ran, from `response_metadata.precontext`. */
export function precontextNames(message: { response_metadata: Record<string, unknown> }): string[] {
  const pc = message.response_metadata.precontext as Array<{ name?: string }> | undefined;
  return (pc ?? []).map((p) => p?.name).filter((n): n is string => typeof n === "string");
}

export function text(message: { content: unknown }): string {
  return typeof message.content === "string" ? message.content : JSON.stringify(message.content);
}

export const lower = (s: unknown): string => String(s).toLowerCase();
