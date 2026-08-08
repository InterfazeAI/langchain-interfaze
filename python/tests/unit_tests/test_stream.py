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
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel

from langchain_interfaze import ChatInterfaze
from tests.unit_tests.conftest import (
    PLAIN_STREAM,
    REPEATED_SIDE,
    STREAM_CHUNKS,
    THINK_SPLIT,
    chunk,
    completion,
    mock_json,
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


@respx.mock
def test_unterminated_tag_recovers_text() -> None:
    """A truncated response must not come back silently empty."""
    mock_sse(
        [
            chunk({"content": "<think>never closed and the real answer 42"}),
            chunk({}, "length"),
        ]
    )
    chunks = list(ChatInterfaze(api_key="t").stream([HumanMessage("x")]))
    body = "".join(c.content for c in chunks if isinstance(c.content, str))
    # Truncated mid-<think>: the partial reasoning is metadata, not the answer.
    assert body == ""
    reasoning = [c.additional_kwargs["reasoning"] for c in chunks if c.additional_kwargs.get("reasoning")]
    assert reasoning == ["never closed and the real answer 42"]


@respx.mock
def test_unmatched_tag_survives_non_streaming() -> None:
    """Non-streaming has the whole body: an unmatched tag is prose, not a side channel."""
    mock_json(completion("Wrap your reasoning in <think> tags."))
    res = ChatInterfaze(api_key="t").invoke([HumanMessage("x")])
    assert res.content == "Wrap your reasoning in <think> tags."


@respx.mock
def test_unterminated_tag_mid_text_does_not_duplicate_prefix() -> None:
    mock_sse([chunk({"content": "The answer is 42. <think>because reasons"}), chunk({}, "length")])
    chunks = list(ChatInterfaze(api_key="t").stream([HumanMessage("x")]))
    body = "".join(c.content for c in chunks if isinstance(c.content, str))
    assert body == "The answer is 42. "
    reasoning = [c.additional_kwargs["reasoning"] for c in chunks if c.additional_kwargs.get("reasoning")]
    assert reasoning == ["because reasons"]


@respx.mock
def test_envelope_side_field_kept_alongside_inline() -> None:
    mock_sse(
        [
            chunk({"content": "<think>INLINE</think>Hi"}),
            chunk({}, "stop"),
            {
                "id": "req-test",
                "object": "chat.completion.chunk",
                "created": 1_700_000_000,
                "model": "interfaze-beta",
                "choices": [],
                "reasoning": "ENVELOPE",
            },
        ]
    )
    merged = None
    for c in ChatInterfaze(api_key="t").stream([HumanMessage("x")]):
        merged = c if merged is None else merged + c
    assert merged is not None
    assert "ENVELOPE" in str(merged.additional_kwargs["reasoning"])
    assert "INLINE" in str(merged.additional_kwargs["reasoning"])


@respx.mock
def test_empty_precontext_still_surfaces() -> None:
    mock_json(completion("hi", precontext=[]))
    md = ChatInterfaze(api_key="t").invoke([HumanMessage("x")]).response_metadata
    assert md["precontext"] == []


@respx.mock
def test_generation_text_matches_stripped_content() -> None:
    mock_json(completion("<think>SECRET</think>The answer is 42"))
    res = ChatInterfaze(api_key="t").generate([[HumanMessage("x")]])
    assert res.generations[0][0].text == "The answer is 42"


@respx.mock
def test_tail_recovered_when_visible_text_starts_with_whitespace() -> None:
    mock_sse(
        [
            chunk({"content": "<think>why</think>\nThe sky is"}),
            chunk({"content": " blue because <precontext>"}),
        ]
    )
    body = "".join(
        c.content
        for c in ChatInterfaze(api_key="t").stream([HumanMessage("x")])
        if isinstance(c.content, str)
    )
    # the response completed, so an unmatched tag is prose and survives
    assert body == "\nThe sky is blue because <precontext>"


@respx.mock
def test_empty_envelope_value_does_not_block_inline_payload() -> None:
    mock_json(completion('<precontext>[{"name":"ocr"}]</precontext>The sky is blue.', precontext=[]))
    md = ChatInterfaze(api_key="t").invoke([HumanMessage("x")]).response_metadata
    assert md["precontext"] == [{"name": "ocr"}]


@respx.mock
def test_truncated_precontext_is_not_shown_as_content() -> None:
    """A half-written <precontext> is internal tool JSON, never the answer."""
    mock_sse(
        [
            chunk({"content": "Total is "}),
            chunk({"content": '<precontext>[{"name":"ocr","result":{"ssn":"123-45-6789"'}, "length"),
        ]
    )
    body = "".join(
        c.content
        for c in ChatInterfaze(api_key="t").stream([HumanMessage("x")])
        if isinstance(c.content, str)
    )
    assert "precontext" not in body
    assert "123-45-6789" not in body
    assert body == "Total is "


@respx.mock
def test_invoke_truncated_precontext_is_not_content() -> None:
    """The truncation rule applies to invoke(), not just streaming."""
    mock_json(completion('Total is <precontext>[{"ssn":"123-45-6789"', finish_reason="length"))
    res = ChatInterfaze(api_key="t").invoke([HumanMessage("x")])
    assert "123-45-6789" not in str(res.content)


@respx.mock
def test_invoke_truncated_think_becomes_reasoning() -> None:
    mock_json(completion("<think>SSN 123-45-6789 so", finish_reason="length"))
    res = ChatInterfaze(api_key="t").invoke([HumanMessage("x")])
    assert "123-45-6789" not in str(res.content)
    assert "123-45-6789" in str(res.response_metadata["reasoning"])


@respx.mock
def test_completed_response_keeps_prose_that_mentions_a_tag() -> None:
    """An unmatched tag in a finished response is prose, not a side channel."""
    mock_json(completion("Wrap metadata in <precontext> tags, then continue."))
    res = ChatInterfaze(api_key="t").invoke([HumanMessage("x")])
    assert res.content == "Wrap metadata in <precontext> tags, then continue."


@respx.mock
def test_final_side_chunk_stamps_model_provider() -> None:
    mock_sse(THINK_SPLIT)
    chunks = list(ChatInterfaze(api_key="t").stream([HumanMessage("x")]))
    # core appends its own empty chunk_position="last" sentinel; every chunk we emit is stamped
    ours = [c for c in chunks if c.chunk_position != "last"]
    assert {c.response_metadata.get("model_provider") for c in ours} == {"interfaze"}


@respx.mock
def test_structured_output_streams_and_keeps_side_fields() -> None:
    """with_structured_output streams through beta.chat.completions, which nests every
    frame under "chunk" and omits `role` on the completion it assembles."""

    class Ans(BaseModel):
        answer: str

    # conftest's chunk() omits `role`, exactly as interfaze does after the first delta
    frames = [chunk({"content": '{"answer":'}), chunk({"content": '"blue"}'}), chunk({}, "stop")]
    frames[-1]["precontext"] = [{"name": "ocr", "output": "x"}]
    mock_sse(frames)
    model = ChatInterfaze(api_key="t").with_structured_output(Ans, include_raw=True)
    out = [c for c in model.stream([HumanMessage("x")])]
    assert any(c.get("parsed") == Ans(answer="blue") for c in out)
    raw = next(c["raw"] for c in out if c.get("raw"))
    assert raw.additional_kwargs["precontext"] == [{"name": "ocr", "output": "x"}]


@respx.mock
def test_roleless_deltas_still_produce_ai_message_chunks() -> None:
    frames = [chunk({"content": "hi"}), chunk({}, "stop")]
    mock_sse(frames)
    chunks = list(ChatInterfaze(api_key="t").stream([HumanMessage("x")]))
    assert all(isinstance(c, AIMessageChunk) for c in chunks)
    assert "".join(c.content for c in chunks) == "hi"  # ty:ignore[no-matching-overload]


_EVENT_FRAMES = [
    chunk({"content": "<think>r</think>Hi"}),
    {
        "id": "req-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "interfaze-beta",
        "choices": [],
        "vcache": True,
        "precontext": [{"name": "ocr"}],
    },
    chunk({}, "length"),
]


@respx.mock
def test_astream_events_strips_tags_and_keeps_metadata() -> None:
    """The js package needs a hand-written _streamChatModelEvents to reach this; python
    gets it from the shared chunk path. Both must agree on what a v3 consumer sees."""

    async def run() -> Any:
        mock_sse(_EVENT_FRAMES)
        async for ev in ChatInterfaze(api_key="t").astream_events([HumanMessage("x")], version="v2"):
            if ev["event"] == "on_chat_model_end":
                return ev["data"]["output"]
        raise AssertionError("no on_chat_model_end event")

    out = asyncio.run(run())
    assert out.content == "Hi"
    assert out.response_metadata["finish_reason"] == "length"
    assert out.response_metadata["vcache"] is True
    assert out.response_metadata["precontext"] == [{"name": "ocr"}]
    assert out.response_metadata["reasoning"] == "r"


@respx.mock
def test_invoke_falls_through_an_empty_think_to_recovered_reasoning() -> None:
    mock_json(completion("<think></think>visible<think>partial reasoning", finish_reason="length"))
    res = ChatInterfaze(api_key="t").invoke([HumanMessage("x")])
    assert res.content == "visible"
    assert res.response_metadata["reasoning"] == "partial reasoning"
