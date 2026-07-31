# Interfaze LangChain SDK

The official [LangChain](https://js.langchain.com) integration for [Interfaze](https://interfaze.ai)

[Docs](https://interfaze.ai/docs) · [limits](https://interfaze.ai/docs/limits) · [pricing](https://interfaze.ai/pricing) · [dashboard](https://interfaze.ai) · [Python SDK](https://github.com/InterfazeAI/interfaze-python) · [TypeScript / JavaScript SDK](https://github.com/InterfazeAI/interfaze-js)

## Install

```bash
npm install @interfaze/langchain
```

`@langchain/openai`, `@langchain/core`, and `interfaze` are peer dependencies - `@interfaze/langchain` builds `ChatInterfaze` on top of them. The structured-output and tool examples below use `zod` for schemas (`npm install zod`); it's an optional peer.

## Setup

```ts
import { ChatInterfaze } from "@interfaze/langchain";

const llm = new ChatInterfaze({ apiKey: "sk_..." }); // or set INTERFAZE_API_KEY and call new ChatInterfaze()
```

`ChatInterfaze` is a standard LangChain chat model, so the usual fields (`temperature`, `maxTokens`, `timeout`, …) are forwarded; `configuration.baseURL` and `model` default to the Interfaze endpoint and `interfaze-beta`.

## Your first request

Extract structured data from an ID. Interfaze runs OCR for you, `withStructuredOutput` returns your schema, and the raw OCR lands on `response_metadata.precontext` - pass `includeRaw: true` to keep both:

```ts
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { z } from "zod";

const IdCard = z.object({
  first_name: z.string(),
  last_name: z.string(),
  dob: z.string().describe("Date of birth on the ID"),
  licence_number: z.string(),
});

const out = await llm.withStructuredOutput(IdCard, { includeRaw: true }).invoke([
  new HumanMessage({
    content: [
      { type: "text", text: "Extract the details from this ID." },
      { type: "image_url", image_url: { url: "https://r2public.jigsawstack.com/interfaze/examples/id.jpg" } },
    ],
  }),
]);

console.log(out.parsed); // { first_name: "IVÁN ICHET", … }
console.log((out.raw as AIMessage).response_metadata.precontext); // the raw OCR that produced it
```

## Precontext

Interfaze returns fields a plain chat model would drop. `ChatInterfaze` surfaces them on both `response_metadata` and `additional_kwargs`:

```ts
const res = await llm.invoke("Which US public companies reported earnings today?");

res.response_metadata.precontext; // raw output of any tool Interfaze ran (OCR / web / scrape / …)
res.response_metadata.reasoning; // reasoning text (with reasoningEffort and no schema)
res.response_metadata.vcache; // whether the semantic cache was hit
```

## Chat

```ts
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const res = await llm.invoke([new SystemMessage("You are concise."), new HumanMessage("Which US public companies reported earnings today?")]);

res.content; // a web search backs the answer here
```

Pass a plain string for a one-off (`llm.invoke("…")`), or a message list for multi-turn.

### Streaming

Stream the reply as it's generated; the inline `<think>`/`<precontext>` side-channels are stripped from the streamed content:

```ts
for await (const chunk of await llm.stream("Summarize this week's top AI research and cite your sources.")) {
  process.stdout.write(typeof chunk.content === "string" ? chunk.content : "");
}
```

### Structured output

`withStructuredOutput` takes a zod schema (or JSON schema) and returns instances:

```ts
import { z } from "zod";

const Receipt = z.object({
  merchant: z.string(),
  total: z.number(),
});

const structured = llm.withStructuredOutput(Receipt);
await structured.invoke([
  new HumanMessage({
    content: [
      { type: "text", text: "Extract this receipt." },
      { type: "image_url", image_url: { url: "https://jigsawstack.com/preview/vocr-example.jpg" } },
    ],
  }),
]); // -> { merchant: "Walmart", total: 144.02 }
```

Pass `{ includeRaw: true }` to also get the underlying `AIMessage` (and its `precontext`).

### Tools and function calling

Bind tools with `bindTools`, then read `tool_calls` off the response:

```ts
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const getWeather = tool(
  async ({ city }) => {
    return `Sunny in ${city}`;
  },
  {
    name: "get_weather",
    description: "Get the current weather for a city.",
    schema: z.object({ city: z.string() }),
  }
);

const res = await llm.bindTools([getWeather]).invoke("What's the weather in Tokyo?");
res.tool_calls; // [{ name: "get_weather", args: { city: "Tokyo" }, id: ... }]
```

## Reasoning

Pass `reasoningEffort` as a call option (`"low"` / `"medium"` / `"high"`, …); the reasoning text comes back on `response_metadata.reasoning`:

```ts
const res = await llm.invoke("Which region should we launch in first, and why?", { reasoningEffort: "high" });
res.response_metadata.reasoning;
```

Use `.withConfig({ reasoningEffort: "high" })` to apply it to every call on a model instance instead of passing it per-invoke.

## Multimodal Inputs

Images, audio, PDFs, and CSV use standard LangChain content parts, by URL or base64:

```ts
await llm.invoke([
  new HumanMessage({
    content: [
      { type: "text", text: "Summarize this document." },
      { type: "file", file: { filename: "paper.pdf", file_data: "https://arxiv.org/pdf/1706.03762" } },
    ],
  }),
]);
```

Video rides on an Interfaze `file` part via a `{ type: "video", ... }` block:

```ts
await llm.invoke([
  new HumanMessage({
    content: [
      { type: "text", text: "What happens in this clip?" },
      { type: "video", url: "https://…/clip.mp4" },
    ] as never,
  }),
]);
```

> A video block accepts `url`, `base64` (with an optional `mime_type`), or `file_id`, plus an optional `extras: { filename: … }`.

## Async and batch

`invoke`, `stream`, and `batch` are all async already - there is no separate sync API to reach for:

```ts
await llm.invoke("Hello");

for await (const chunk of await llm.stream("Hello")) {
  process.stdout.write(typeof chunk.content === "string" ? chunk.content : "");
}

await llm.batch(["Summarize A", "Summarize B", "Summarize C"]);
```

`batch` fans the calls out concurrently.

## Chains (LCEL)

Chain `ChatInterfaze` like any other LangChain runnable, via `.pipe()`:

```ts
import { ChatPromptTemplate } from "@langchain/core/prompts";

const chain = ChatPromptTemplate.fromTemplate("Translate to {lang}: {text}").pipe(llm);
await chain.invoke({ lang: "French", text: "Hello" });
```

## Feeding precontext

Pass precomputed tool output to skip Interfaze's internal tool run:

```ts
const llm = new ChatInterfaze({ precontext: [{ name: "ocr", result: { extracted_text: "..." } }] });
```

## Tasks and guardrails

`ChatInterfaze` is a chat model. For the one-shot `tasks.*` helpers ([run_task](https://interfaze.ai/docs/run-tasks)) and `guard` safety codes, use the core [`interfaze`](https://github.com/InterfazeAI/interfaze-js) client directly.

## Errors

```ts
import { BadRequestError, InterfazeError, RateLimitError } from "interfaze";
```

`ChatInterfaze` raises `InterfazeError` for client-side problems (a missing API key). Everything else is an `APIError` subclass carrying `status` and `code` - `BadRequestError` (400), `AuthenticationError` (401), `RateLimitError` (429), and so on.

## Capabilities

| Use case                                    | Entry point                         |
| ------------------------------------------- | ----------------------------------- |
| [Chat](#chat)                               | `invoke` / `stream`                 |
| [Structured output](#structured-output)     | `withStructuredOutput(schema)`      |
| [Tools](#tools-and-function-calling)        | `bindTools([...])`                  |
| [Reasoning](#reasoning)                     | `reasoningEffort` call option       |
| [Multimodal inputs](#multimodal-inputs)     | content parts + `{ type: "video" }` |
| [Precontext](#precontext)                   | `response_metadata.precontext`      |
| [Async and batch](#async-and-batch)         | `invoke` / `stream` / `batch`       |
| [Chains](#chains-lcel)                      | LCEL (`.pipe()`)                    |
| [Feed precontext](#feeding-precontext)      | `new ChatInterfaze({ precontext })` |
| [Tasks / guardrails](#tasks-and-guardrails) | core `interfaze` client             |

## License

MIT
