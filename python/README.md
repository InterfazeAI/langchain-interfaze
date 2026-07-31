# Interfaze LangChain SDK

The official [LangChain](https://python.langchain.com) integration for [Interfaze](https://interfaze.ai)

[Docs](https://interfaze.ai/docs) · [limits](https://interfaze.ai/docs/limits) · [pricing](https://interfaze.ai/pricing) · [dashboard](https://interfaze.ai) · [Python SDK](https://github.com/InterfazeAI/interfaze-python) · [TypeScript / JavaScript SDK](https://github.com/InterfazeAI/interfaze-js)

## Install

```bash
pip install langchain-interfaze
```

This pulls in the `interfaze` client and the LangChain packages it builds on.

## Setup

```python
from langchain_interfaze import ChatInterfaze

llm = ChatInterfaze(api_key="sk_...")  # or set INTERFAZE_API_KEY and call ChatInterfaze()
```

`ChatInterfaze` is a standard LangChain chat model, so the usual keywords (`temperature`, `max_tokens`, `timeout`, `reasoning_effort`, …) are forwarded; `base_url` and `model` default to the Interfaze endpoint and `interfaze-beta`.

## Your first request

Extract structured data from an ID. Interfaze runs OCR for you, `with_structured_output` returns your schema, and the raw OCR lands on `response_metadata["precontext"]` - pass `include_raw=True` to keep both:

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

## Precontext

Interfaze returns fields a plain chat model would drop. `ChatInterfaze` surfaces them on both `response_metadata` and `additional_kwargs`:

```python
res = llm.invoke("Which US public companies reported earnings today?")

res.response_metadata.get("precontext")  # raw output of any tool Interfaze ran (OCR / web / scrape / …)
res.response_metadata.get("reasoning")   # reasoning text (with reasoning_effort and no schema)
res.response_metadata.get("vcache")      # whether the semantic cache was hit
```

## Chat

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

Pass a plain string for a one-off (`llm.invoke("…")`), or a message list for multi-turn.

### Streaming

Stream the reply as it's generated; the inline `<think>`/`<precontext>` side-channels are stripped from the streamed content:

```python
for chunk in llm.stream("Summarize this week's top AI research and cite your sources."):
    print(chunk.content, end="", flush=True)
```

### Structured output

`with_structured_output` takes a Pydantic model (or JSON schema) and returns instances:

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

Pass `include_raw=True` to also get the underlying `AIMessage` (and its `precontext`).

### Tools and function calling

Bind tools with `bind_tools`, then read `tool_calls` off the response:

```python
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    ...


res = llm.bind_tools([get_weather]).invoke("What's the weather in Tokyo?")
res.tool_calls  # [{"name": "get_weather", "args": {"city": "Tokyo"}, "id": ...}]
```

## Reasoning

Set `reasoning_effort`; the reasoning text comes back on `response_metadata["reasoning"]`:

```python
llm = ChatInterfaze(reasoning_effort="high")  # also "on" / "off" / "auto"; or llm.bind(reasoning_effort="high")

res = llm.invoke("Which region should we launch in first, and why?")
res.response_metadata.get("reasoning")
```

## Multimodal Inputs

Images, audio, PDFs, and CSV use standard LangChain content parts, by URL or base64:

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

Video rides on an Interfaze `file` part via a `{"type": "video", ...}` block:

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

> A video block accepts `url`, `base64` (with an optional `mime_type`), or `file_id`, plus an optional `extras={"filename": …}`.

## Async and batch

Every call has an async twin, and `batch` fans out concurrently:

```python
await llm.ainvoke("Hello")

async for chunk in llm.astream("Hello"):
    print(chunk.content, end="")

llm.batch(["Summarize A", "Summarize B", "Summarize C"])
```

## Chains (LCEL)

Chain `ChatInterfaze` like any other LangChain runnable:

```python
from langchain_core.prompts import ChatPromptTemplate

chain = ChatPromptTemplate.from_template("Translate to {lang}: {text}") | llm
chain.invoke({"lang": "French", "text": "Hello"})
```

## Feeding precontext

Pass precomputed tool output to skip Interfaze's internal tool run:

```python
llm = ChatInterfaze(precontext=[{"name": "ocr", "result": {"extracted_text": "..."}}])
```

## Tasks and guardrails

`ChatInterfaze` is a chat model. For the one-shot `tasks.*` helpers ([run_task](https://interfaze.ai/docs/run-tasks)) and `guard` safety codes, use the core [`interfaze`](https://github.com/InterfazeAI/interfaze-python) client directly.

## Errors

```python
from interfaze import BadRequestError, InterfazeError, RateLimitError
```

`ChatInterfaze` raises `InterfazeError` for client-side problems (a missing API key). Everything else is an `APIError` subclass carrying `status_code` and `code` - `BadRequestError` (400), `AuthenticationError` (401), `RateLimitError` (429), and so on.

## Capabilities

| Use case                                    | Entry point                            |
| ------------------------------------------- | -------------------------------------- |
| [Chat](#chat)                               | `invoke` / `stream`                    |
| [Structured output](#structured-output)     | `with_structured_output(Model)`        |
| [Tools](#tools-and-function-calling)        | `bind_tools([...])`                    |
| [Reasoning](#reasoning)                     | `reasoning_effort`                     |
| [Multimodal inputs](#multimodal-inputs)     | content parts + `{"type": "video"}`    |
| [Precontext](#precontext)                   | `response_metadata["precontext"]`      |
| [Async and batch](#async-and-batch)         | `ainvoke` / `astream` / `batch`        |
| [Chains](#chains-lcel)                      | LCEL (`\|`)                            |
| [Feed precontext](#feeding-precontext)      | `ChatInterfaze(precontext=[...])`      |
| [Tasks / guardrails](#tasks-and-guardrails) | core `interfaze` client                |

## License

MIT
