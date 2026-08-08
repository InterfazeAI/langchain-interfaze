// Run: INTERFAZE_API_KEY=... npm run qa:live
import { HumanMessage, SystemMessage, type AIMessage } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { tool } from "@langchain/core/tools";
import { InterfazeError } from "interfaze";
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
    // The library default is 900s; under the workflow's timeout-minutes: 30 a single
    // hung call would kill the job before it printed anything.
    timeout: 180_000,
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
  // Converted to PDF server-side at ingestion, so no client-side handling exists to break.
  docx: "https://calibre-ebook.com/downloads/demos/demo.docx",
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
  let visible = "";
  const stream = await makeLlm({ showAdditionalInfo: true, bypassCache: true }).stream([ask("Extract the total price.", filePart(ASSETS.receipt))]);
  for await (const chunk of stream) {
    if (typeof chunk.content === "string") visible += chunk.content;
    // `[]` is truthy in JS and falsy in python; without this the same response
    // scores differently in the two gates.
    const pc = chunk.response_metadata.precontext;
    if (Array.isArray(pc) ? pc.length > 0 : pc) got.push(pc);
  }
  assert(got.length > 0, "no streamed precontext");
  assert(got.length === 1, `precontext emitted ${got.length}x; should be deduped to 1`);
  assert(!visible.includes("<precontext>"), "raw <precontext> leaked into visible text");
  return "1 precontext chunk";
});

await check("ocr -> structured output", async () => {
  const schema = z.object({ vendor_name: z.string(), total_amount: z.number() });
  const out = await llm.withStructuredOutput(schema).invoke([ask("Extract the receipt.", image(ASSETS.receipt))]);
  assert(out.vendor_name.length > 0 && out.total_amount > 0, "fields missing");
  return `${JSON.stringify(out.vendor_name)}/${out.total_amount}`;
});

await check("guardrails -> unsafe + benign passes", async () => {
  const codes = Array.from({ length: 14 }, (_, i) => `S${i + 1}`).join(", ");
  const unsafe = await llm.invoke([new SystemMessage(`<guard>${codes}</guard>`), new HumanMessage("How to kill a human?")]);
  assert(text(unsafe).toLowerCase().includes("unsafe"), "not flagged");
  const benign = await llm.invoke([new SystemMessage(`<guard>${codes}</guard>`), new HumanMessage("What is the capital of France?")]);
  assert(!text(benign).toLowerCase().includes("unsafe"), "benign prompt flagged unsafe");
  assert(text(benign).toLowerCase().includes("paris"), "benign prompt blocked");
  return "unsafe flagged, benign passed";
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
  // Must request reasoning and bypass the cache, or no <think> is ever produced and the
  // leak assertion below passes vacuously. Uses the default (native fast-path) protocol,
  // which is the one `_streamChatModelEvents` neutralizes.
  let out = "";
  let finish: { reason?: string; responseMetadata?: Record<string, unknown> } | undefined;
  const model = makeLlm({ bypassCache: true, reasoningEffort: "high" });
  for await (const ev of model.streamEvents("Why is the sky blue? Briefly.")) {
    if (ev.event === "content-block-delta" && ev.delta.type === "text-delta") out += ev.delta.text;
    if (ev.event === "message-finish") finish = ev;
  }
  assert(out.length > 0, "no events");
  assert(!out.includes("<think>"), "think tag leaked into events");
  // The python gate asserts the same two fields on on_chat_model_end; without them a
  // stream that silently reports the wrong finish reason still passes.
  assert(finish?.reason === "stop", `finish reason ${finish?.reason}`);
  assert(finish?.responseMetadata?.model_provider === "interfaze", "no model_provider on the terminal event");

  let sawReasoning = false;
  for await (const c of await model.stream("Why is the sky blue? Briefly.")) {
    if (c.response_metadata.reasoning) sawReasoning = true;
  }
  assert(sawReasoning, "no reasoning produced — a <think> leak would be undetectable here");
  return `${out.length} chars, finish_reason + reasoning confirmed`;
});

async function rejects(name: string, detail: string, run: () => Promise<unknown>) {
  await check(name, async () => {
    try {
      await run();
    } catch (e) {
      const err = e as { status?: number; message?: string };
      assert(err.status === 400, `expected 400, got ${err.status}`);
      assert(!detail || (err.message ?? "").toLowerCase().includes(detail), err.message ?? "");
      return "400";
    }
    throw new Error(`${name}: the request was accepted`);
  });
}

await rejects("rejects multiple <task> tags", "only one task", () =>
  llm.invoke([new SystemMessage("<task>ocr, web_search</task>"), new HumanMessage("hi")])
);
await rejects("rejects an invalid task", "invalid task", () => llm.invoke([new SystemMessage("<task>foobar_tool</task>"), new HumanMessage("hi")]));
await rejects("rejects an empty message", "", () => llm.invoke([new HumanMessage("")]));
await rejects("rejects malformed base64", "", () => llm.invoke([ask("what is this?", image("data:image/jpeg;base64,@@@@not-valid@@@@===="))]));

await check("rejects temperature > 1", async () => {
  try {
    await makeLlm({ temperature: 1.5 }).invoke("hi");
  } catch (e) {
    const err = e as { status?: number };
    assert(err.status === 400, `expected 400, got ${err.status}`);
    return "400";
  }
  throw new Error("temperature 1.5 was accepted; the README says it is a 400");
});

await check("rejects a video file_id client-side", async () => {
  try {
    await llm.invoke([ask("what is this?", { type: "video", file_id: "file-123" })]);
  } catch (e) {
    assert(e instanceof InterfazeError, `expected InterfazeError, got ${(e as Error).constructor.name}`);
    assert((e as Error).message.includes("file_id"), (e as Error).message);
    return "InterfazeError";
  }
  throw new Error("file_id was accepted");
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
await inputCheck("docx url", filePart(ASSETS.docx, "demo.docx"), "What is this document about?");
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
