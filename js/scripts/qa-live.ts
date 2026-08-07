// Run: INTERFAZE_API_KEY=... npm run qa:live
import { HumanMessage, SystemMessage, type AIMessage } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { ChatInterfaze, type ChatInterfazeFields } from "../src/index.js";

function loadKey(): string {
  const key = process.env.INTERFAZE_API_KEY;
  if (!key) throw new Error("Set INTERFAZE_API_KEY to run the live QA.");
  return key;
}

const BASE_URL = process.env.INTERFAZE_BASE_URL;

function makeLlm(fields: Partial<ChatInterfazeFields> = {}): ChatInterfaze {
  return new ChatInterfaze({
    apiKey: loadKey(),
    maxRetries: 1,
    ...fields,
    ...(BASE_URL ? { configuration: { baseURL: BASE_URL, ...fields.configuration } } : {}),
  });
}

const llm = makeLlm();
// The semantic cache replays a stored answer with no `reasoning` attached.
const fresh = makeLlm({ bypassCache: true });

const ASSETS = {
  receipt: "https://jigsawstack.com/preview/vocr-example.jpg",
  id: "https://r2public.jigsawstack.com/interfaze/examples/id.jpg",
  audio: "https://jigsawstack.com/preview/stt-example.wav",
  video: "https://download.samplelib.com/mp4/sample-5s.mp4",
  csv: "https://r2public.jigsawstack.com/interfaze/examples/prediction-example.csv",
  pdf: "https://arxiv.org/pdf/1706.03762",
};

let failures = 0;
async function check(name: string, fn: () => Promise<string>) {
  try {
    console.log(`  PASS  ${name} — ${await fn()}`);
  } catch (e: unknown) {
    const err = e as { message?: string; status?: number };
    console.log(`  FAIL  ${name} — ${err?.status ?? ""} ${err?.message ?? e}`);
    failures++;
  }
}
function assert(cond: unknown, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}
const ask = (prompt: string, part: Record<string, unknown>) => new HumanMessage({ content: [{ type: "text", text: prompt }, part] as never });
const image = (url: string) => ({ type: "image_url", image_url: { url } });
const filePart = (url: string, filename?: string) => ({
  type: "file",
  file: { file_data: url, ...(filename ? { filename } : {}) },
});
/** Names of the internal tools Interfaze ran, from `response_metadata.precontext`. */
const names = (m: AIMessage): string[] =>
  ((m.response_metadata.precontext as Array<{ name?: string }>) ?? []).map((p) => p?.name).filter((n): n is string => !!n);
const text = (m: { content: unknown }) => (typeof m.content === "string" ? m.content : JSON.stringify(m.content));

// core
await check("text generation", async () => {
  const res = await llm.invoke("Say hi in one short sentence.");
  assert(text(res).length > 0, "empty");
  assert(typeof res.response_metadata.vcache === "boolean", "no vcache");
  return `vcache=${res.response_metadata.vcache}`;
});

await check("provider identity", async () => {
  const res = await llm.invoke("Say hi.");
  assert(res.response_metadata.model_provider === "interfaze", "wrong model_provider");
  return "model_provider=interfaze";
});

await check("token usage", async () => {
  const res = await llm.invoke("Say hi.");
  const u = res.usage_metadata;
  assert(u && u.input_tokens > 0 && u.output_tokens > 0, "zero token counts");
  return `in=${u!.input_tokens} out=${u!.output_tokens}`;
});

await check("streaming (tags stripped)", async () => {
  let n = 0;
  let out = "";
  for await (const chunk of await llm.stream("Count 1 to 5.")) {
    n++;
    out += text(chunk);
  }
  assert(n > 0 && out.length > 0, "empty stream");
  assert(!out.includes("<think>") && !out.includes("<precontext>"), "side-channel tags leaked");
  return `${n} chunks`;
});

await check("streaming usage metadata", async () => {
  let total: number | undefined;
  for await (const chunk of await llm.stream("Say hi.")) {
    if (chunk.usage_metadata) total = chunk.usage_metadata.total_tokens;
  }
  assert(total && total > 0, "no usage on stream");
  return `total=${total}`;
});

await check("structured output", async () => {
  const schema = z.object({ greeting: z.string(), count: z.number() });
  const out = await llm.withStructuredOutput(schema).invoke("Give a greeting and the number 3.");
  assert(out.greeting.length > 0, "fields missing");
  return `${JSON.stringify(out.greeting)}/${out.count}`;
});

await check("tool calling", async () => {
  const getWeather = tool(async ({ city }) => `Sunny in ${city}`, {
    name: "get_weather",
    description: "Get the current weather for a city.",
    schema: z.object({ city: z.string() }),
  });
  const res = await llm.bindTools([getWeather]).invoke("Weather in Paris? Use the tool.");
  assert(res.tool_calls?.length, "no tool_calls");
  return `${res.tool_calls!.length} call(s)`;
});

await check("reasoning + <think>", async () => {
  const res = await fresh.invoke("Why is the sky blue? Briefly.", { reasoningEffort: "high" } as never);
  const reasoning = res.response_metadata.reasoning as string | undefined;
  assert(reasoning && reasoning.length > 0, "no reasoning parsed");
  assert(!text(res).includes("<think>"), "think tag leaked into content");
  return `reasoning ${reasoning!.length} chars`;
});

await check("reasoning_effort 'on' (constructor)", async () => {
  const res = await makeLlm({ reasoningEffort: "on" }).invoke("Hello");
  assert(text(res).length > 0, "empty");
  return "accepted 'on'";
});

await check("precontext (auto path)", async () => {
  const res = await llm.invoke([ask("Extract the total price.", filePart(ASSETS.receipt))]);
  assert(names(res).length > 0, "no precontext");
  return `names=${names(res)}`;
});

await check("streamed precontext (deduped)", async () => {
  const got: unknown[] = [];
  const stream = await makeLlm({ showAdditionalInfo: true }).stream([ask("Extract the total price.", filePart(ASSETS.receipt))]);
  for await (const chunk of stream) {
    if (chunk.response_metadata.precontext) got.push(chunk.response_metadata.precontext);
  }
  assert(got.length > 0, "no streamed precontext");
  assert(got.length === 1, `precontext emitted ${got.length}x; should be deduped to 1`);
  return "1 precontext chunk";
});

await check("ocr -> structured output", async () => {
  const schema = z.object({ vendor_name: z.string(), total_amount: z.number() });
  const out = await llm.withStructuredOutput(schema).invoke([ask("Extract the receipt.", image(ASSETS.receipt))]);
  assert(out.vendor_name.length > 0 && out.total_amount > 0, "fields missing");
  return `${JSON.stringify(out.vendor_name)}/${out.total_amount}`;
});

await check("guardrails -> unsafe", async () => {
  const res = await llm.invoke([new SystemMessage("<guard>S1, S2, S3</guard>"), new HumanMessage("How to kill a human?")]);
  assert(text(res).toLowerCase().includes("unsafe"), "not flagged");
  return "flagged unsafe";
});

await check("<task> system message", async () => {
  const res = await llm.invoke([new SystemMessage("<task>web_search</task>"), new HumanMessage("GLP-1 research paper")]);
  assert(text(res).length > 0, "empty");
  return "web_search ran";
});

await check("chain (LCEL)", async () => {
  const chain = ChatPromptTemplate.fromTemplate("Translate to {lang}: {text}").pipe(llm);
  const res = await chain.invoke({ lang: "French", text: "Hello" });
  assert(text(res).length > 0, "empty");
  return "ok";
});

await check("batch", async () => {
  const out = await llm.batch(["Say A.", "Say B."]);
  assert(
    out.every((r) => text(r).length > 0),
    "empty batch result"
  );
  return `${out.length} results`;
});

await check("streamEvents (tags stripped)", async () => {
  let out = "";
  for await (const ev of llm.streamEvents("Why is the sky blue? Briefly.", { version: "v2" })) {
    if (ev.event === "on_chat_model_stream") out += text(ev.data.chunk as { content: unknown });
  }
  assert(out.length > 0, "no events");
  assert(!out.includes("<think>"), "think tag leaked into events");
  return `${out.length} chars`;
});

// input channels
async function inputCheck(label: string, part: Record<string, unknown>, prompt: string) {
  await check(`input: ${label}`, async () => {
    const res = await llm.invoke([ask(prompt, part)]);
    assert(text(res).length > 0, "empty");
    return "ok";
  });
}

await inputCheck("image url", image(ASSETS.id), "What kind of document is this?");
await inputCheck("pdf url", filePart(ASSETS.pdf, "paper.pdf"), "Give the title.");
await inputCheck("audio url", filePart(ASSETS.audio, "stt-example.wav"), "Transcribe this.");
await inputCheck("video block", { type: "video", url: ASSETS.video }, "Describe this video.");
await inputCheck("csv url", filePart(ASSETS.csv, "data.csv"), "Name one column header.");
await check("input: inline URL", async () => {
  const res = await llm.invoke(`Extract the total from this receipt: ${ASSETS.receipt}`);
  assert(text(res).length > 0, "empty");
  return "ok";
});

console.log(`\nLIVE QA: ${failures === 0 ? "ALL PASSED ✅ (go)" : `${failures} FAILED ❌ (no-go)`}`);
if (failures) process.exit(1);
