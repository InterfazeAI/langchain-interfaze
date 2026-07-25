# langchain-interfaze

The [LangChain](https://python.langchain.com) SDK for [Interfaze](https://interfaze.ai).

## Learn more

- [interfaze.ai](https://interfaze.ai) - dashboard and API keys.
- [Interfaze Python SDK](https://github.com/InterfazeAI/interfaze-python) - the core (non-LangChain) client.
- [LangChain docs](https://python.langchain.com) - chat models, tools, and LCEL.

## Install

```bash
pip install langchain-interfaze
```

This pulls in the `interfaze` and `langchain-openai` packages it builds on.

## Setup

Get an API key from the [Interfaze dashboard](https://interfaze.ai), then:

```python
from langchain_interfaze import ChatInterfaze

llm = ChatInterfaze(api_key="sk_...")  # or set INTERFAZE_API_KEY and call ChatInterfaze()
```

`base_url` and `model` are optional (they default to the Interfaze endpoint and `interfaze-beta`); any other `ChatOpenAI` keyword (`temperature`, `max_tokens`, `timeout`, `reasoning_effort`, …) is forwarded.

## Usage

Invoke:

```python
res = llm.invoke("Write a haiku about deterministic AI.")
print(res.content)
print(res.response_metadata.get("vcache"))  # Interfaze extra: semantic-cache hit
```

Messages:

```python
from langchain_core.messages import HumanMessage, SystemMessage

llm.invoke(
    [
        SystemMessage("You are concise."),
        HumanMessage("Summarize the latest AI news."),
    ]
)
```

Streaming - inline `<think>` / `<precontext>` side-channels are stripped from the streamed content:

```python
for chunk in llm.stream("Tell me a story."):
    print(chunk.content, end="")
```

Async and batch:

```python
await llm.ainvoke("Hello")
async for chunk in llm.astream("Hello"):
    print(chunk.content, end="")
llm.batch(["Summarize A", "Summarize B", "Summarize C"])
```

Tool calling:

```python
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    ...


res = llm.bind_tools([get_weather]).invoke("What's the weather in Tokyo?")
print(res.tool_calls)
```

Structured output:

```python
from pydantic import BaseModel


class Weather(BaseModel):
    city: str
    temp_c: float


structured = llm.with_structured_output(Weather)
structured.invoke("Weather in Tokyo?")  # -> Weather(city="Tokyo", temp_c=...)
```

Multimodal - images, audio, and PDFs use standard content parts; video rides on an Interfaze file part
via a `{"type": "video", ...}` block:

```python
from langchain_core.messages import HumanMessage

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

LCEL - chain it like any other LangChain runnable:

```python
from langchain_core.prompts import ChatPromptTemplate

chain = ChatPromptTemplate.from_template("Translate to {lang}: {text}") | llm
chain.invoke({"lang": "French", "text": "Hello"})
```

## Interfaze extras

Interfaze returns fields a stock `ChatOpenAI` discards. `ChatInterfaze` surfaces them on both
`response_metadata` and `additional_kwargs`:

| Field        | Meaning                                                        |
| ------------ | -------------------------------------------------------------  |
| `precontext` | Raw output of any internal tools Interfaze ran (OCR/web/…).    |
| `reasoning`  | Reasoning text (with `reasoning_effort="high"` and no schema). |
| `vcache`     | Whether the semantic cache was hit.                            |

```python
res = llm.invoke("...")
res.response_metadata.get("precontext")
res.response_metadata.get("reasoning")
res.response_metadata.get("vcache")
```

Feed precomputed tool output to skip Interfaze's internal tool run:

```python
llm = ChatInterfaze(precontext=[{"name": "ocr", "result": {"extracted_text": "..."}}])
```

## License

MIT
