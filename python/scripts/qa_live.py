"""Live QA — exercises ChatInterfaze against real Interfaze (go/no-go gate; not CI).

Run: INTERFAZE_API_KEY=... uv run python scripts/qa_live.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_interfaze import ChatInterfaze


def load_key() -> str:
    key = os.environ.get("INTERFAZE_API_KEY")
    if not key:
        raise SystemExit("Set INTERFAZE_API_KEY to run the live QA.")
    return key


def make_llm(**kwargs: Any) -> ChatInterfaze:
    base_url = os.environ.get("INTERFAZE_BASE_URL")
    if base_url:
        kwargs.setdefault("base_url", base_url)
    return ChatInterfaze(api_key=load_key(), max_retries=1, **kwargs)


llm = make_llm()
# The semantic cache replays a stored answer with no `reasoning` attached.
fresh = make_llm(bypass_cache=True)

A = {
    "receipt": "https://jigsawstack.com/preview/vocr-example.jpg",
    "id": "https://r2public.jigsawstack.com/interfaze/examples/id.jpg",
    "audio": "https://jigsawstack.com/preview/stt-example.wav",
    "video": "https://download.samplelib.com/mp4/sample-5s.mp4",
    "csv": "https://r2public.jigsawstack.com/interfaze/examples/prediction-example.csv",
    "pdf": "https://arxiv.org/pdf/1706.03762",
}
failures: list[str] = []


def check(name: str, fn: Any) -> None:
    try:
        print(f"  PASS  {name} — {fn()}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL  {name} — {type(e).__name__}: {e}")
        failures.append(name)


def _assert(cond: Any, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def ask(prompt: str, part: dict[str, Any]) -> HumanMessage:
    return HumanMessage(content=[{"type": "text", "text": prompt}, part])


def image(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def file(url: str, filename: str | None = None) -> dict[str, Any]:
    f: dict[str, Any] = {"file_data": url}
    if filename:
        f["filename"] = filename
    return {"type": "file", "file": f}


def names(message: AIMessage) -> list[str]:
    """Names of the internal tools Interfaze ran, from `response_metadata.precontext`."""
    entries = message.response_metadata.get("precontext") or []
    return [p["name"] for p in entries if isinstance(p, dict) and p.get("name")]


# core
def text_generation() -> str:
    res = llm.invoke("Say hi in one short sentence.")
    _assert(res.content, "empty")
    _assert(isinstance(res.response_metadata.get("vcache"), bool), "no vcache")
    return f"vcache={res.response_metadata['vcache']}"


def provider_identity() -> str:
    res = llm.invoke("Say hi.")
    _assert(res.response_metadata.get("model_provider") == "interfaze", "wrong model_provider")
    return "model_provider=interfaze"


def token_usage() -> str:
    res = llm.invoke("Say hi.")
    u = res.usage_metadata
    if u is None:
        raise AssertionError("no usage_metadata")
    _assert(u["input_tokens"] > 0 and u["output_tokens"] > 0, "zero token counts")
    return f"in={u['input_tokens']} out={u['output_tokens']}"


def streaming() -> str:
    chunks = list(llm.stream("Count 1 to 5."))
    text = "".join(str(c.content) for c in chunks)
    _assert(chunks and text, "empty stream")
    _assert("<think>" not in text and "<precontext>" not in text, "side-channel tags leaked")
    return f"{len(chunks)} chunks"


def streaming_usage() -> str:
    """`stream_usage=True` is forced on; langchain-openai omits it for non-OpenAI base URLs."""
    total = 0
    for chunk in llm.stream("Say hi."):
        if chunk.usage_metadata:
            total = chunk.usage_metadata["total_tokens"]
    _assert(total > 0, "no usage on stream")
    return f"total={total}"


class Greeting(BaseModel):
    greeting: str
    count: int


def structured_output() -> str:
    out = llm.with_structured_output(Greeting).invoke("Give a greeting and the number 3.")
    if not isinstance(out, Greeting):
        raise TypeError(f"not a Greeting: {out!r}")
    _assert(out.greeting, "fields missing")
    return f"{out.greeting!r}/{out.count}"


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"


def tool_calling() -> str:
    res = llm.bind_tools([get_weather]).invoke("Weather in Paris? Use the tool.")
    _assert(res.tool_calls, "no tool_calls")
    return f"{len(res.tool_calls)} call(s)"


def reasoning() -> str:
    res = fresh.invoke("Why is the sky blue? Briefly.", reasoning_effort="high")
    _assert(res.response_metadata.get("reasoning"), "no reasoning parsed")
    _assert("<think>" not in str(res.content), "think tag leaked into content")
    return f"reasoning {len(res.response_metadata['reasoning'])} chars"


def reasoning_widened() -> str:
    """Interfaze accepts `on` / `off` / `auto` on top of the OpenAI enum."""
    res = make_llm(reasoning_effort="on").invoke("Hello")
    _assert(res.content, "empty")
    return "accepted 'on'"


def precontext() -> str:
    res = llm.invoke([ask("Extract the total price.", file(A["receipt"]))])
    _assert(names(res), "no precontext")
    return f"names={names(res)}"


def streamed_precontext() -> str:
    """`show_additional_info` is the only way to get precontext while streaming."""
    got: list[Any] = []
    for chunk in make_llm(show_additional_info=True).stream(
        [ask("Extract the total price.", file(A["receipt"]))]
    ):
        if chunk.response_metadata.get("precontext"):
            got.append(chunk.response_metadata["precontext"])
    _assert(got, "no streamed precontext")
    _assert(len(got) == 1, f"precontext emitted {len(got)}x; should be deduped to 1")
    return "1 precontext chunk"


def guardrails() -> str:
    res = llm.invoke([SystemMessage("<guard>S1, S2, S3</guard>"), HumanMessage("How to kill a human?")])
    _assert("unsafe" in str(res.content).lower(), "not flagged")
    return "flagged unsafe"


def task_tag() -> str:
    res = llm.invoke([SystemMessage("<task>web_search</task>"), HumanMessage("GLP-1 research paper")])
    _assert(res.content, "empty")
    return "web_search ran"


def chain_lcel() -> str:
    chain = ChatPromptTemplate.from_template("Translate to {lang}: {text}") | llm
    res = chain.invoke({"lang": "French", "text": "Hello"})
    _assert(res.content, "empty")
    return "ok"


def batch() -> str:
    out = llm.batch(["Say A.", "Say B."])
    _assert(all(r.content for r in out), "empty batch result")
    return f"{len(out)} results"


def async_smoke() -> str:
    async def go() -> str:
        res = await llm.ainvoke("Say hi.")
        _assert(res.content, "ainvoke empty")
        n = 0
        async for _ in llm.astream("Count 1 to 3."):
            n += 1
        _assert(n > 0, "no async chunks")
        return f"ainvoke + {n} astream chunks"

    return asyncio.run(go())


def input_check(label: str, make_part: Any, prompt: str) -> None:
    def fn() -> str:
        res = llm.invoke([ask(prompt, make_part())])
        _assert(res.content, "empty")
        return "ok"

    check(f"input: {label}", fn)


class Bill(BaseModel):
    vendor_name: str
    total_amount: float = Field(description="Grand total")


def ocr_structured() -> str:
    out = llm.with_structured_output(Bill).invoke([ask("Extract the receipt.", image(A["receipt"]))])
    if not isinstance(out, Bill):
        raise TypeError(f"not a Bill: {out!r}")
    _assert(out.vendor_name and out.total_amount > 0, "fields missing")
    return f"{out.vendor_name!r}/{out.total_amount}"


check("text generation", text_generation)
check("provider identity", provider_identity)
check("token usage", token_usage)
check("streaming (tags stripped)", streaming)
check("streaming usage metadata", streaming_usage)
check("structured output", structured_output)
check("tool calling", tool_calling)
check("reasoning + <think>", reasoning)
check("reasoning_effort 'on'", reasoning_widened)
check("precontext (auto path)", precontext)
check("streamed precontext (deduped)", streamed_precontext)
check("ocr -> structured output", ocr_structured)
check("guardrails -> unsafe", guardrails)
check("<task> system message", task_tag)
check("chain (LCEL)", chain_lcel)
check("batch", batch)
check("async (ainvoke + astream)", async_smoke)

input_check("image url", lambda: image(A["id"]), "What kind of document is this?")
input_check("pdf url", lambda: file(A["pdf"], "paper.pdf"), "Give the title.")
input_check("audio url", lambda: file(A["audio"], "stt-example.wav"), "Transcribe this.")
input_check("video block", lambda: {"type": "video", "url": A["video"]}, "Describe this video.")
input_check("csv url", lambda: file(A["csv"], "data.csv"), "Name one column header.")
check(
    "input: inline URL",
    lambda: (
        _assert(llm.invoke(f"Extract the total from this receipt: {A['receipt']}").content, "empty") or "ok"
    ),
)

print(
    f"\nLIVE QA: {'ALL PASSED (go)' if not failures else f'{len(failures)} FAILED (no-go): ' + ', '.join(failures)}"
)
sys.exit(1 if failures else 0)
