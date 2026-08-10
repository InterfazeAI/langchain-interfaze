# Interfaze LangChain SDK

The official [LangChain](https://www.langchain.com) integration for [Interfaze](https://interfaze.ai), for both **Python** (`interfaze-langchain`) and **TypeScript / JavaScript** (`@interfaze/langchain`).

[Docs](https://interfaze.ai/docs) · [limits](https://interfaze.ai/docs/limits) · [pricing](https://interfaze.ai/pricing) · [dashboard](https://interfaze.ai) · [Python SDK](https://github.com/InterfazeAI/interfaze-python) · [TypeScript / JavaScript SDK](https://github.com/InterfazeAI/interfaze-js)

`ChatInterfaze` is a standard LangChain chat model with the same behavior in both languages. Every section below shows Python and TypeScript side by side. Source lives in [`python/`](./python) and [`js/`](./js).

## Install

Python:

```bash
pip install interfaze-langchain
```

TypeScript / JavaScript:

```bash
npm install @interfaze/langchain
```

The TS structured-output and tool examples use `zod` for schemas (`npm install zod`); it's an optional peer.

## Setup

Python:

```python
from interfaze_langchain import ChatInterfaze

llm = ChatInterfaze(api_key="sk_...")  # or set INTERFAZE_API_KEY and call ChatInterfaze()
```

TypeScript:

```ts
import { ChatInterfaze } from "@interfaze/langchain";

const llm = new ChatInterfaze({ apiKey: "sk_..." }); // or set INTERFAZE_API_KEY and call new ChatInterfaze()
```

`ChatInterfaze` is a standard LangChain chat model, so the usual options (`temperature`, `max_tokens` / `maxTokens`, `timeout`, `reasoning_effort` / `reasoningEffort`, …) are forwarded; the base URL (`base_url` / `configuration.baseURL`) and `model` default to the Interfaze endpoint and `interfaze-beta`.

## Your first request

Extract structured data from an ID. Interfaze runs OCR for you, structured output returns your schema, and the raw OCR lands on `response_metadata.precontext` — keep both with `include_raw` / `includeRaw`:

Python:

```python
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field


class IdCard(BaseModel):
    first_name: str
    last_name: str
    dob: str = Field(description="Date of birth on the ID")
    licence_number: str


out = llm.with_structured_output(IdCard, include_raw=True).invoke(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "Extract the details from this ID."},
                {"type": "image_url", "image_url": {"url": "https://r2public.jigsawstack.com/interfaze/examples/id.jpg"}},
            ]
        )
    ]
)

print(out["parsed"])  # IdCard(first_name="IVÁN ICHET", …)
print(out["raw"].response_metadata.get("precontext"))  # the raw OCR that produced it
```

TypeScript:

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

Python:

```python
res = llm.invoke("Which US public companies reported earnings today?")

res.response_metadata.get("precontext")  # raw output of any tool Interfaze ran (OCR / web / scrape / …)
res.response_metadata.get("reasoning")   # reasoning text (with reasoning_effort and no schema)
res.response_metadata.get("vcache")      # whether the semantic cache was hit
```

TypeScript:

```ts
const res = await llm.invoke("Which US public companies reported earnings today?");

res.response_metadata.precontext; // raw output of any tool Interfaze ran (OCR / web / scrape / …)
res.response_metadata.reasoning; // reasoning text (with reasoningEffort and no schema)
res.response_metadata.vcache; // whether the semantic cache was hit
```

## Chat

Pass a plain string for a one-off, or a message list for multi-turn.

Python:

```python
from langchain_core.messages import HumanMessage, SystemMessage

res = llm.invoke(
    [
        SystemMessage("You are concise."),
        HumanMessage("Which US public companies reported earnings today?"),
    ]
)

res.content  # a web search backs the answer here
```

TypeScript:

```ts
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

const res = await llm.invoke([new SystemMessage("You are concise."), new HumanMessage("Which US public companies reported earnings today?")]);

res.content; // a web search backs the answer here
```

### Streaming

Stream the reply as it's generated; the inline `<think>`/`<precontext>` side-channels are stripped from the streamed content:

Python:

```python
for chunk in llm.stream("Summarize this week's top AI research and cite your sources."):
    print(chunk.content, end="", flush=True)
```

TypeScript:

```ts
for await (const chunk of await llm.stream("Summarize this week's top AI research and cite your sources.")) {
  process.stdout.write(typeof chunk.content === "string" ? chunk.content : "");
}
```

### Structured output

Takes a Pydantic model / zod schema (or JSON schema) and returns instances. Pass `include_raw` / `includeRaw` to also get the underlying `AIMessage` (and its `precontext`).

Python:

```python
from pydantic import BaseModel


class Receipt(BaseModel):
    merchant: str
    total: float


structured = llm.with_structured_output(Receipt)
structured.invoke(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "Extract this receipt."},
                {"type": "image_url", "image_url": {"url": "https://jigsawstack.com/preview/vocr-example.jpg"}},
            ]
        )
    ]
)  # -> Receipt(merchant="Walmart", total=144.02)
```

TypeScript:

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

### Tools and function calling

Bind tools with `bind_tools` / `bindTools`, then read `tool_calls` off the response:

Python:

```python
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    ...


res = llm.bind_tools([get_weather]).invoke("What's the weather in Tokyo?")
res.tool_calls  # [{"name": "get_weather", "args": {"city": "Tokyo"}, "id": ...}]
```

TypeScript:

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

The reasoning text comes back on `response_metadata.reasoning`.

Python — set `reasoning_effort`:

```python
llm = ChatInterfaze(reasoning_effort="high")  # also "on" / "off" / "auto"; or llm.bind(reasoning_effort="high")

res = llm.invoke("Which region should we launch in first, and why?")
res.response_metadata.get("reasoning")
```

TypeScript — pass `reasoningEffort` as a call option (`"low"` / `"medium"` / `"high"`, …), or `.withConfig({ reasoningEffort: "high" })` to apply it to every call:

```ts
const res = await llm.invoke("Which region should we launch in first, and why?", { reasoningEffort: "high" });
res.response_metadata.reasoning;
```

## Multimodal Inputs

Images, audio, PDFs, Word documents (`.docx`), and CSV use standard LangChain content parts, by URL or base64:

Python:

```python
from langchain_core.messages import HumanMessage

llm.invoke(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "Summarize this document."},
                {"type": "file", "file": {"filename": "paper.pdf", "file_data": "https://arxiv.org/pdf/1706.03762"}},
            ]
        )
    ]
)
```

TypeScript:

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

Video rides on an Interfaze `file` part via a `{"type": "video", ...}` block:

Python:

```python
llm.invoke(
    [
        HumanMessage(
            content=[
                {"type": "text", "text": "What happens in this clip?"},
                {"type": "video", "url": "https://…/clip.mp4"},
            ]
        )
    ]
)
```

TypeScript:

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

> A video block accepts `url` or `base64` (with an optional `mime_type`), plus an optional `extras` `{"filename": …}`.
> The container mime type is inferred from the URL extension when you don't pass one. Interfaze has no file store, so `file_id` is not supported.

## Async and batch

Python — every call has an async twin, and `batch` fans out concurrently:

```python
await llm.ainvoke("Hello")

async for chunk in llm.astream("Hello"):
    print(chunk.content, end="")

llm.batch(["Summarize A", "Summarize B", "Summarize C"])
```

TypeScript — `invoke`, `stream`, and `batch` are all async already (no separate sync API); `batch` fans out concurrently:

```ts
await llm.invoke("Hello");

for await (const chunk of await llm.stream("Hello")) {
  process.stdout.write(typeof chunk.content === "string" ? chunk.content : "");
}

await llm.batch(["Summarize A", "Summarize B", "Summarize C"]);
```

## Chains (LCEL)

Chain `ChatInterfaze` like any other LangChain runnable.

Python (`|`):

```python
from langchain_core.prompts import ChatPromptTemplate

chain = ChatPromptTemplate.from_template("Translate to {lang}: {text}") | llm
chain.invoke({"lang": "French", "text": "Hello"})
```

TypeScript (`.pipe()`):

```ts
import { ChatPromptTemplate } from "@langchain/core/prompts";

const chain = ChatPromptTemplate.fromTemplate("Translate to {lang}: {text}").pipe(llm);
await chain.invoke({ lang: "French", text: "Hello" });
```

## Client options

Set router, cache, and streaming behavior once on the client:

Python:

```python
llm = ChatInterfaze(
    show_additional_info=True,  # emit inline <precontext> while streaming
    bypass_cache=True,          # skip the semantic cache
    bypass_moa=True,            # skip the mixture-of-architecture router
)
```

TypeScript:

```ts
const llm = new ChatInterfaze({
  showAdditionalInfo: true, // emit inline <precontext> while streaming
  bypassCache: true, // skip the semantic cache
  bypassMoA: true, // skip the mixture-of-architecture router
});
```

`showAdditionalInfo` / `show_additional_info` is the only way to get `precontext` **while streaming** — non-streaming responses always carry it. `bypass_cache` matters when you need a fresh generation: a cache hit replays the stored answer, which has no `reasoning` attached.

The request timeout defaults to **900 s**, because a single call may run OCR, a web search or a transcription inline. Pass `timeout` to change it.

## Tasks and guardrails

Interfaze reads `<task>` and `<guard>` tags from the **first system message**, so both work through a plain LangChain `SystemMessage`:

Python:

```python
from langchain_core.messages import HumanMessage, SystemMessage

llm.invoke([SystemMessage("<task>web_search</task>"), HumanMessage("GLP-1 research paper")])
llm.invoke([SystemMessage("<guard>S1, S2, S3</guard>"), HumanMessage("How to kill a human?")])  # -> "unsafe S1"
```

TypeScript:

```ts
await llm.invoke([new SystemMessage("<task>web_search</task>"), new HumanMessage("GLP-1 research paper")]);
await llm.invoke([new SystemMessage("<guard>S1, S2, S3</guard>"), new HumanMessage("How to kill a human?")]); // -> "unsafe S1"
```

One task at a time, from `ocr`, `object_detection`, `gui_detection`, `web_search`, `scraper`, `translate`, `speech_to_text`, `forecast`, `classification`. A task cannot be combined with a non-empty structured-output schema.

For the one-shot `tasks.*` helpers ([run_task](https://interfaze.ai/docs/run-tasks)), use the core `interfaze` client directly ([Python](https://github.com/InterfazeAI/interfaze-python) · [TypeScript / JavaScript](https://github.com/InterfazeAI/interfaze-js)).

## Server limits

`ChatInterfaze` forwards standard LangChain options, but validates only the subset supported by Interfaze:

| Option                                | Accepted                                                            |
| ------------------------------------- | ------------------------------------------------------------------- |
| `temperature`                         | `0`–`1` (values above `1` are a `400`)                              |
| `max_tokens` / `maxTokens`            | `1`–`32000`                                                         |
| `reasoning_effort` / `reasoningEffort`| `minimal`, `low`, `medium`, `high`, plus `on` / `off` / `auto`      |
| `tool_choice`                         | ignored — the router always picks                                   |
| `stop`, `n`, `seed`, `logprobs`       | ignored                                                             |

## Errors

Python:

```python
from interfaze import BadRequestError, InterfazeError, RateLimitError
```

TypeScript:

```ts
import { BadRequestError, InterfazeError, RateLimitError } from "interfaze";
```

`ChatInterfaze` raises `InterfazeError` for client-side problems (a missing API key). Everything else is an `APIError` subclass carrying a status code (`status_code` in Python, `status` in TS) and `code` — `BadRequestError` (400), `AuthenticationError` (401), `RateLimitError` (429), and so on.

## Capabilities

| Use case                                    | Python                            | TypeScript / JavaScript             |
| ------------------------------------------- | --------------------------------- | ----------------------------------- |
| [Chat](#chat)                               | `invoke` / `stream`               | `invoke` / `stream`                 |
| [Structured output](#structured-output)     | `with_structured_output(Model)`   | `withStructuredOutput(schema)`      |
| [Tools](#tools-and-function-calling)        | `bind_tools([...])`               | `bindTools([...])`                  |
| [Reasoning](#reasoning)                     | `reasoning_effort`                | `reasoningEffort` call option       |
| [Multimodal inputs](#multimodal-inputs)     | content parts + `{"type":"video"}`| content parts + `{ type: "video" }` |
| [Precontext](#precontext)                   | `response_metadata["precontext"]` | `response_metadata.precontext`      |
| [Async and batch](#async-and-batch)         | `ainvoke` / `astream` / `batch`   | `invoke` / `stream` / `batch`       |
| [Chains](#chains-lcel)                      | LCEL (`\|`)                       | LCEL (`.pipe()`)                    |
| [Client options](#client-options)           | `bypass_cache=True`, …            | `bypassCache: true`, …              |
| [Tasks / guardrails](#tasks-and-guardrails) | `SystemMessage("<task>…</task>")` | `new SystemMessage("<task>…")`      |

## License

MIT
