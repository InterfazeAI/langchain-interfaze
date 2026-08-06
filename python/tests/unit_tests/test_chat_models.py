from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
import respx
from interfaze import INTERFAZE_BASE_URL, INTERFAZE_MODEL, InterfazeError
from langchain_core.callbacks import BaseCallbackHandler, CallbackManager
from langchain_core.messages import HumanMessage

from langchain_interfaze import ChatInterfaze

CHAT_URL = "https://api.interfaze.ai/v1/chat/completions"
VIDEO_URL = "https://download.samplelib.com/mp4/sample-5s.mp4"

_USAGE = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


def completion(content: Any = "Hi!", *, finish_reason: str = "stop", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "req-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": "interfaze-beta",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": None},
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": _USAGE,
        "vcache": False,
    }
    body.update(extra)
    return body


def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "req-test",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "interfaze-beta",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _sse_bytes(chunks: list[dict[str, Any]]) -> bytes:
    return ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()


def mock_json(body: dict[str, Any]) -> respx.Route:
    return respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json=body))


def mock_sse(chunks: list[dict[str, Any]]) -> respx.Route:
    return respx.post(CHAT_URL).mock(
        return_value=httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=_sse_bytes(chunks)
        )
    )


def last_body(route: respx.Route) -> dict[str, Any]:
    return json.loads(route.calls.last.request.content)


BASIC = completion("Hi!")
CUSTOM_FIELDS = completion(
    "Hello there",
    precontext=[{"name": "ocr", "result": {"extracted_text": "x"}}],
    reasoning="because reasons",
    vcache=True,
)
STREAM_CHUNKS: list[dict[str, Any]] = [
    _chunk({"content": '<precontext>[{"name":"ocr","result":{"extracted_text":"x"}}]</precontext>'}),
    _chunk({"content": "Total "}),
    _chunk({"content": "is $12.34"}),
    _chunk({}, finish_reason="stop"),
]
THINK_SPLIT: list[dict[str, Any]] = [
    _chunk({"content": "<th"}),
    _chunk({"content": "ink>Rayleigh scat"}),
    _chunk({"content": "tering.</think>The sky "}),
    _chunk({"content": "is blue."}),
    _chunk({}, finish_reason="stop"),
]


# defaults
def test_defaults_point_at_interfaze() -> None:
    model = ChatInterfaze(api_key="t")
    assert model.openai_api_base == INTERFAZE_BASE_URL
    assert model.model_name == INTERFAZE_MODEL


def test_defaults_overridable() -> None:
    model = ChatInterfaze(api_key="t", base_url="https://example.com/v1", model="other-model")
    assert model.openai_api_base == "https://example.com/v1"
    assert model.model_name == "other-model"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERFAZE_API_KEY", raising=False)
    with pytest.raises(InterfazeError, match="Missing API key"):
        ChatInterfaze()


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERFAZE_API_KEY", "env-key")
    model = ChatInterfaze()
    assert model.openai_api_key is not None


# custom response fields
@respx.mock
def test_custom_response_fields_surfaced() -> None:
    mock_json(CUSTOM_FIELDS)
    model = ChatInterfaze(api_key="t")
    result = model.invoke([HumanMessage("hi")])
    assert result.response_metadata["precontext"] == [{"name": "ocr", "result": {"extracted_text": "x"}}]
    assert result.response_metadata["reasoning"] == "because reasons"
    assert result.response_metadata["vcache"] is True
    assert result.additional_kwargs["precontext"] == [{"name": "ocr", "result": {"extracted_text": "x"}}]
    assert result.additional_kwargs["reasoning"] == "because reasons"
    assert result.additional_kwargs["vcache"] is True


@respx.mock
def test_response_without_precontext_or_reasoning_unaffected() -> None:
    mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    result = model.invoke([HumanMessage("hi")])
    assert "precontext" not in result.response_metadata
    assert "reasoning" not in result.response_metadata
    assert result.response_metadata["vcache"] is False
    assert result.content == "Hi!"


# provider identity
def test_provider_identity() -> None:
    model = ChatInterfaze(api_key="t")
    assert model._llm_type == "interfaze-chat"
    assert model._get_ls_params()["ls_provider"] == "interfaze"
    assert model.lc_secrets == {"openai_api_key": "INTERFAZE_API_KEY"}
    assert model.get_lc_namespace() == ["langchain_interfaze", "chat_models"]
    assert model.metadata is not None
    assert "langchain-interfaze" in model.metadata["lc_versions"]


@respx.mock
def test_model_provider_stamped_on_response() -> None:
    mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    assert model.invoke([HumanMessage("hi")]).response_metadata["model_provider"] == "interfaze"


def test_defaults_to_long_timeout_but_respects_override() -> None:
    assert ChatInterfaze(api_key="t").request_timeout == 900.0
    assert ChatInterfaze(api_key="t", timeout=30).request_timeout == 30


def test_never_routes_to_the_responses_api() -> None:
    # `reasoning=` would otherwise flip ChatOpenAI over to /v1/responses.
    assert ChatInterfaze(api_key="t", reasoning={"summary": "auto"}).use_responses_api is False


# control-plane headers
def test_control_headers() -> None:
    model = ChatInterfaze(
        api_key="t",
        show_additional_info=True,
        bypass_moa=True,
        bypass_cache=True,
        admin_key="adm",
        default_headers={"x-custom": "1"},
    )
    assert model.default_headers == {
        "x-custom": "1",
        "x-show-additional-info": "true",
        "x-interfaze-bypass-moa": "true",
        "x-interfaze-bypass-cache": "true",
        "x-admin-key": "adm",
    }


def test_no_control_headers_by_default() -> None:
    assert ChatInterfaze(api_key="t").default_headers is None


@respx.mock
def test_streaming_asks_for_usage() -> None:
    # langchain-openai only auto-enables this for OpenAI's own base URL.
    route = mock_sse([_chunk({"content": "hi"}), _chunk({}, finish_reason="stop")])
    list(ChatInterfaze(api_key="t").stream([HumanMessage("hi")]))
    assert last_body(route)["stream_options"] == {"include_usage": True}


# video content blocks
@respx.mock
def test_video_block_converted_to_file_part() -> None:
    route = mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "what happens in this clip?"},
            {"type": "video", "url": VIDEO_URL},
        ]
    )
    model.invoke([message])  # must not raise
    body = last_body(route)
    content = body["messages"][-1]["content"]
    assert {"type": "file", "file": {"file_data": VIDEO_URL, "format": "video/mp4"}} in content


@respx.mock
def test_video_block_url_without_known_extension_omits_format() -> None:
    route = mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    model.invoke([HumanMessage(content=[{"type": "video", "url": "https://example.com/clip"}])])
    assert last_body(route)["messages"][-1]["content"][0]["file"] == {"file_data": "https://example.com/clip"}


def test_video_block_file_id_raises() -> None:
    model = ChatInterfaze(api_key="t")
    with pytest.raises(InterfazeError, match="file_id"):
        model.invoke([HumanMessage(content=[{"type": "video", "file_id": "file-123"}])])


@respx.mock
def test_video_block_base64_converted_to_file_part() -> None:
    route = mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    message = HumanMessage(content=[{"type": "video", "base64": "AAAA", "mime_type": "video/mp4"}])
    model.invoke([message])
    body = last_body(route)
    content = body["messages"][-1]["content"]
    assert content[0]["type"] == "file"
    assert content[0]["file"]["file_data"] == "data:video/mp4;base64,AAAA"


# inline tag stripping (streaming)
@respx.mock
def test_streaming_strips_inline_tags_and_carries_precontext() -> None:
    mock_sse(STREAM_CHUNKS)
    model = ChatInterfaze(api_key="t")
    chunks = list(model.stream([HumanMessage("x")]))
    text = "".join(c.content for c in chunks)  # ty:ignore[no-matching-overload]
    assert "<precontext>" not in text
    assert text == "Total is $12.34"
    precontext_chunks = [c for c in chunks if c.additional_kwargs.get("precontext")]
    assert precontext_chunks
    assert precontext_chunks[0].additional_kwargs["precontext"][0]["name"] == "ocr"


@respx.mock
def test_streaming_recovers_reasoning_split_across_chunks() -> None:
    mock_sse(THINK_SPLIT)
    model = ChatInterfaze(api_key="t")
    chunks = list(model.stream([HumanMessage("x")]))
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


# async
@respx.mock
def test_async_invoke_surfaces_side_fields() -> None:
    mock_json(CUSTOM_FIELDS)
    model = ChatInterfaze(api_key="t")

    async def go() -> Any:
        return await model.ainvoke([HumanMessage("hi")])

    result = asyncio.run(go())
    assert result.response_metadata["precontext"][0]["name"] == "ocr"
    assert result.response_metadata["vcache"] is True


# inline tag stripping (non-streaming)
@respx.mock
def test_non_streaming_strips_inline_tags() -> None:
    content = (
        "<think>Rayleigh scattering.</think>"
        '<precontext>[{"name": "ocr", "result": {"x": 1}}]</precontext>'
        "The sky is blue."
    )
    mock_json(completion(content))
    model = ChatInterfaze(api_key="t")
    result = model.invoke([HumanMessage("why is the sky blue?")])
    assert result.content == "The sky is blue."
    assert result.response_metadata["reasoning"] == "Rayleigh scattering."
    assert result.response_metadata["precontext"] == [{"name": "ocr", "result": {"x": 1}}]


@respx.mock
def test_video_block_forwards_filename() -> None:
    route = mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    model.invoke(
        [HumanMessage(content=[{"type": "video", "url": VIDEO_URL, "extras": {"filename": "clip.mp4"}}])]
    )
    file = last_body(route)["messages"][-1]["content"][0]["file"]
    assert file["file_data"] == VIDEO_URL
    assert file["filename"] == "clip.mp4"


def test_video_block_missing_source_raises() -> None:
    model = ChatInterfaze(api_key="t")
    with pytest.raises(InterfazeError, match="requires one of"):
        model.invoke([HumanMessage(content=[{"type": "video"}])])


# streaming with no side channels
@respx.mock
def test_streaming_plain_content_emits_no_side_channel_chunk() -> None:
    mock_sse([_chunk({"content": "Hello "}), _chunk({"content": "world"}), _chunk({}, finish_reason="stop")])
    model = ChatInterfaze(api_key="t")
    chunks = list(model.stream([HumanMessage("hi")]))
    assert "".join(c.content for c in chunks) == "Hello world"  # ty:ignore[no-matching-overload]
    assert not any(
        c.additional_kwargs.get("precontext") or c.additional_kwargs.get("reasoning") for c in chunks
    )


# token callbacks must never see the raw side-channel tags. Core's own stream() calls
# `_stream` without a run_manager, but the v2 protocol path passes one straight through,
# and ChatOpenAI fires on_llm_new_token before yielding — hence the explicit check here.
@respx.mock
def test_run_manager_tokens_are_filtered() -> None:
    mock_sse(THINK_SPLIT)
    model = ChatInterfaze(api_key="t")
    seen: list[str] = []

    class Tap(BaseCallbackHandler):
        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            seen.append(token)

    manager = CallbackManager.configure(inheritable_callbacks=[Tap()])
    run_manager = manager.on_chat_model_start({}, [[HumanMessage("x")]])[0]
    list(model._stream([HumanMessage("x")], run_manager=run_manager))
    assert "<think>" not in "".join(seen)
    assert "".join(seen) == "The sky is blue."


@respx.mock
def test_stream_text_matches_filtered_content() -> None:
    mock_sse(THINK_SPLIT)
    model = ChatInterfaze(api_key="t")
    gens = list(model._stream([HumanMessage("x")]))
    assert all(g.text == g.message.content for g in gens)
