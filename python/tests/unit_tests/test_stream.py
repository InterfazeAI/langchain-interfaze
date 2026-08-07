from __future__ import annotations

import asyncio
from typing import Any

import respx
from langchain_core.callbacks import (
    AsyncCallbackHandler,
    AsyncCallbackManager,
    BaseCallbackHandler,
    CallbackManager,
)
from langchain_core.messages import HumanMessage

from langchain_interfaze import ChatInterfaze
from tests.unit_tests.conftest import (
    PLAIN_STREAM,
    REPEATED_SIDE,
    STREAM_CHUNKS,
    THINK_SPLIT,
    chunk,
    mock_sse,
)


@respx.mock
def test_streaming_strips_inline_tags_and_carries_precontext() -> None:
    mock_sse(STREAM_CHUNKS)
    chunks = list(ChatInterfaze(api_key="t").stream([HumanMessage("x")]))
    text = "".join(c.content for c in chunks)  # ty:ignore[no-matching-overload]
    assert "<precontext>" not in text
    assert text == "Total is $12.34"
    carriers = [c for c in chunks if c.additional_kwargs.get("precontext")]
    assert carriers
    assert carriers[0].additional_kwargs["precontext"][0]["name"] == "ocr"


@respx.mock
def test_streaming_recovers_reasoning_split_across_chunks() -> None:
    mock_sse(THINK_SPLIT)
    chunks = list(ChatInterfaze(api_key="t").stream([HumanMessage("x")]))
    text = "".join(c.content for c in chunks)  # ty:ignore[no-matching-overload]
    assert "<think>" not in text and text == "The sky is blue."
    reasoning = [c for c in chunks if c.additional_kwargs.get("reasoning")]
    assert reasoning and reasoning[0].additional_kwargs["reasoning"] == "Rayleigh scattering."


@respx.mock
def test_async_streaming_recovers_reasoning_split_across_chunks() -> None:
    mock_sse(THINK_SPLIT)
    model = ChatInterfaze(api_key="t")

    async def go() -> list[Any]:
        return [c async for c in model.astream([HumanMessage("x")])]

    chunks = asyncio.run(go())
    text = "".join(c.content for c in chunks)
    assert "<think>" not in text and text == "The sky is blue."
    reasoning = [c for c in chunks if c.additional_kwargs.get("reasoning")]
    assert reasoning and reasoning[0].additional_kwargs["reasoning"] == "Rayleigh scattering."


@respx.mock
def test_streaming_plain_content_emits_no_side_channel_chunk() -> None:
    mock_sse(PLAIN_STREAM)
    chunks = list(ChatInterfaze(api_key="t").stream([HumanMessage("hi")]))
    assert "".join(c.content for c in chunks) == "Hello world"  # ty:ignore[no-matching-overload]
    assert not any(
        c.additional_kwargs.get("precontext") or c.additional_kwargs.get("reasoning") for c in chunks
    )


# Token callbacks must never see the raw side-channel tags. Core's stream() calls `_stream`
# without a run_manager, but the v2 protocol path passes one straight through, and
# ChatOpenAI fires on_llm_new_token before yielding — hence the explicit check here.
@respx.mock
def test_run_manager_tokens_are_filtered() -> None:
    mock_sse(THINK_SPLIT)
    model = ChatInterfaze(api_key="t")
    seen: list[str] = []

    class Tap(BaseCallbackHandler):
        def on_llm_new_token(self, token: str | list[str | dict[str, Any]], **kwargs: Any) -> None:
            seen.append(str(token))

    manager = CallbackManager.configure(inheritable_callbacks=[Tap()])
    run_manager = manager.on_chat_model_start({}, [[HumanMessage("x")]])[0]
    list(model._stream([HumanMessage("x")], run_manager=run_manager))
    assert "<think>" not in "".join(seen)
    assert "".join(seen) == "The sky is blue."


@respx.mock
async def test_async_run_manager_tokens_are_filtered() -> None:
    mock_sse(THINK_SPLIT)
    model = ChatInterfaze(api_key="t")
    seen: list[str] = []

    class Tap(AsyncCallbackHandler):
        async def on_llm_new_token(self, token: str | list[str | dict[str, Any]], **kwargs: Any) -> None:
            seen.append(str(token))

    manager = AsyncCallbackManager.configure(inheritable_callbacks=[Tap()])
    run_manager = (await manager.on_chat_model_start({}, [[HumanMessage("x")]]))[0]
    async for _ in model._astream([HumanMessage("x")], run_manager=run_manager):
        pass
    assert "".join(seen) == "The sky is blue."


@respx.mock
def test_stream_text_matches_filtered_content() -> None:
    mock_sse(THINK_SPLIT)
    gens = list(ChatInterfaze(api_key="t")._stream([HumanMessage("x")]))
    assert all(g.text == g.message.content for g in gens)


@respx.mock
def test_streamed_side_fields_are_applied_once() -> None:
    mock_sse(REPEATED_SIDE)
    chunks = list(ChatInterfaze(api_key="t").stream([HumanMessage("hi")]))
    for key in ("reasoning", "precontext", "vcache"):
        assert sum(key in c.additional_kwargs for c in chunks) == 1, key

    merged = chunks[0]
    for c in chunks[1:]:
        merged = merged + c
    assert merged.additional_kwargs["reasoning"] == "why"
    assert merged.response_metadata["reasoning"] == "why"
    assert merged.additional_kwargs["precontext"] == [{"name": "ocr"}]
    assert merged.additional_kwargs["vcache"] is True
    assert merged.response_metadata["model_provider"] == "interfaze"


@respx.mock
def test_empty_wire_reasoning_does_not_suppress_inline_think() -> None:
    # An empty `reasoning` on the envelope must not mark the field seen, or the genuine
    # inline <think> text recovered from the tail is dropped in its favour.
    mock_sse([chunk({"content": "<think>real</think>ok"}) | {"reasoning": ""}, chunk({}, "stop")])
    out = list(ChatInterfaze(api_key="t").stream([HumanMessage("hi")]))
    got = [c.additional_kwargs["reasoning"] for c in out if "reasoning" in c.additional_kwargs]
    assert got == ["real"]


@respx.mock
def test_vcache_is_deduped_so_it_stays_a_bool() -> None:
    mock_sse(
        [
            chunk({"content": "a"}) | {"vcache": True},
            chunk({"content": "b"}) | {"vcache": False},
            chunk({}, finish_reason="stop"),
        ]
    )
    chunks = list(ChatInterfaze(api_key="t").stream([HumanMessage("hi")]))
    merged = chunks[0]
    for c in chunks[1:]:
        merged = merged + c
    assert merged.additional_kwargs["vcache"] is True


@respx.mock
def test_streamed_reasoning_not_repeated_by_final_chunk() -> None:
    # The tag-derived tail must not re-emit a field the wire already delivered.
    mock_sse([chunk({"content": "<think>why</think>ok"}) | {"reasoning": "why"}, chunk({}, "stop")])
    out = list(ChatInterfaze(api_key="t").stream([HumanMessage("hi")]))
    assert sum("reasoning" in c.additional_kwargs for c in out) == 1
