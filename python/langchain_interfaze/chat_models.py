from __future__ import annotations

import json
import os
import re
from collections.abc import AsyncIterator, Iterator
from typing import Any

from interfaze import (
    INTERFAZE_BASE_URL,
    INTERFAZE_MODEL,
    InterfazeError,
    SideChannelFilter,
    strip_side_channels,
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import SecretStr, model_validator
from typing_extensions import Self

from langchain_interfaze._version import __version__

_PROVIDER = "interfaze"

_DEFAULT_TIMEOUT = 900.0

_HEADER_SHOW_ADDITIONAL_INFO = "x-show-additional-info"
_HEADER_BYPASS_MOA = "x-interfaze-bypass-moa"
_HEADER_BYPASS_CACHE = "x-interfaze-bypass-cache"

_SIDE_FIELDS = ("precontext", "reasoning", "vcache")

_ACCUMULATING_SIDE_FIELDS = ("precontext", "reasoning")

_VIDEO_MIME: dict[str, str] = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "webm": "video/webm",
    "avi": "video/x-msvideo",
    "mkv": "video/x-matroska",
    "3gp": "video/3gpp",
}


def _carries_value(value: Any) -> bool:
    return value is not None and value != ""


def _extract_side_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {k: data[k] for k in _SIDE_FIELDS if _carries_value(data.get(k))}


def _apply_side_fields(message: AIMessage, side: dict[str, Any]) -> None:
    for key, value in side.items():
        message.response_metadata[key] = value
        message.additional_kwargs[key] = value


def _strip_tags(message: AIMessage) -> None:
    if not isinstance(message.content, str) or not (
        "<think>" in message.content or "<precontext>" in message.content
    ):
        return
    text, reasoning, precontext = strip_side_channels(message.content)
    if text != message.content:
        message.content = text
    if reasoning and not message.response_metadata.get("reasoning"):
        message.response_metadata["reasoning"] = reasoning
        message.additional_kwargs["reasoning"] = reasoning
    if precontext and not message.response_metadata.get("precontext"):
        message.response_metadata["precontext"] = precontext
        message.additional_kwargs["precontext"] = precontext


def _video_mime_from_url(url: str) -> str | None:
    base = url.split("?")[0].split("#")[0]
    ext = base.rsplit(".", 1)[-1].lower() if "." in base else ""
    return _VIDEO_MIME.get(ext)


def _convert_video_block(block: dict[str, Any]) -> dict[str, Any]:
    if block.get("file_id") is not None:
        raise InterfazeError("Interfaze cannot resolve a video by 'file_id'. Pass 'url' or 'base64' instead.")
    mime = block.get("mime_type")
    if block.get("url") is not None:
        file: dict[str, Any] = {"file_data": block["url"]}
        mime = mime or _video_mime_from_url(block["url"])
    elif block.get("base64") is not None:
        mime = mime or "video/mp4"
        file = {"file_data": f"data:{mime};base64,{block['base64']}"}
    else:
        raise InterfazeError("Video content block requires one of 'url' or 'base64'.")
    if mime:
        file["format"] = mime
    extras = block.get("extras")
    if isinstance(extras, dict) and extras.get("filename"):
        file["filename"] = extras["filename"]
    return {"type": "file", "file": file}


def _rewrite_video_blocks(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    rewritten = [
        _convert_video_block(block) if isinstance(block, dict) and block.get("type") == "video" else block
        for block in content
    ]
    return rewritten if rewritten != content else content


def _fingerprint(key: str, value: Any) -> str:
    """`precontext`/`reasoning` accumulate, so only an identical payload is a duplicate.

    `vcache` is scalar state — merging two different values would sum them (bool is an int).
    """
    if key not in _ACCUMULATING_SIDE_FIELDS:
        return key
    return f"{key}:{json.dumps(value, sort_keys=True, default=str)}"


def _dedupe_side_fields(message: BaseMessage, seen: set[str]) -> None:
    for key in _SIDE_FIELDS:
        # The response_format branch builds chunks from additional_kwargs alone, with no
        # response_metadata, so reading one map lets those bypass dedupe entirely.
        value = message.response_metadata.get(key)
        if not _carries_value(value):
            value = message.additional_kwargs.get(key)
        if not _carries_value(value):
            continue
        fingerprint = _fingerprint(key, value)
        if fingerprint in seen:
            message.response_metadata.pop(key, None)
            message.additional_kwargs.pop(key, None)
        else:
            seen.add(fingerprint)


def _filter_stream_chunk(
    gen: ChatGenerationChunk, filt: SideChannelFilter, raw: list[str], emitted: list[str]
) -> None:
    message = gen.message
    if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content:
        raw.append(message.content)
        message.content = filt.feed(message.content)
        emitted.append(message.content)
        gen.text = message.content


def _visible_text(raw: str) -> str:
    """`strip_side_channels` trims, which breaks a prefix compare against streamed text."""
    return re.sub(r"<precontext>[\s\S]*?</precontext>", "", re.sub(r"<think>[\s\S]*?</think>", "", raw))


def _missing_tail(raw: str, emitted: str) -> str:
    """The authoritative transcript minus what already streamed."""
    text = _visible_text(raw)
    return text[len(emitted) :] if text.startswith(emitted) else ""


def _final_side_chunk(
    filt: SideChannelFilter, raw: list[str], seen: set[str], emitted: list[str]
) -> ChatGenerationChunk | None:
    tail = filt.flush()
    joined = "".join(raw)
    _, reasoning, precontext = strip_side_channels(joined)
    if not tail:
        tail = _missing_tail(joined, "".join(emitted))
    side: dict[str, Any] = {}
    if reasoning and _fingerprint("reasoning", reasoning) not in seen:
        side["reasoning"] = reasoning
    if precontext and _fingerprint("precontext", precontext) not in seen:
        side["precontext"] = precontext
    if not tail and not side:
        return None
    message = AIMessageChunk(content=tail)
    _apply_side_fields(message, side)
    return ChatGenerationChunk(message=message)


class ChatInterfaze(ChatOpenAI):
    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_interfaze", "chat_models"]

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "INTERFAZE_API_KEY"}

    # A provider-family id, not a model id — `interfaze-beta` reaches tracing and the LLM
    # cache key via `ls_model_name` / `model_name`. Mirrors ChatOpenAI's "openai-chat".
    @property
    def _llm_type(self) -> str:
        return "interfaze"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        show_additional_info: bool = False,
        bypass_moa: bool = False,
        bypass_cache: bool = False,
        default_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        key = api_key or os.environ.get("INTERFAZE_API_KEY")
        if not key:
            raise InterfazeError(
                "Missing API key. Pass ChatInterfaze(api_key=...) or set the INTERFAZE_API_KEY "
                "environment variable."
            )
        headers = dict(default_headers or {})
        if show_additional_info:
            headers[_HEADER_SHOW_ADDITIONAL_INFO] = "true"
        if bypass_moa:
            headers[_HEADER_BYPASS_MOA] = "true"
        if bypass_cache:
            headers[_HEADER_BYPASS_CACHE] = "true"
        if "timeout" not in kwargs and "request_timeout" not in kwargs:
            kwargs["timeout"] = _DEFAULT_TIMEOUT
        kwargs.setdefault("stream_usage", True)
        kwargs["use_responses_api"] = False
        super().__init__(
            api_key=SecretStr(key),
            base_url=base_url or INTERFAZE_BASE_URL,
            model=model or INTERFAZE_MODEL,
            default_headers=headers or None,
            **kwargs,
        )

    # Must be uniquely named: pydantic replaces same-named validators rather than chaining
    # them, so reusing the parent's name would drop its version entry.
    @model_validator(mode="after")
    def _set_interfaze_version(self) -> Self:
        self._add_version("langchain-interfaze", __version__)
        return self

    @property
    def _identifying_params(self) -> dict[str, Any]:
        # Without these, set_llm_cache serves a bypass_cache model the plain model's answer.
        params = {**super()._identifying_params, "_type": self._llm_type}
        if self.default_headers:
            params["interfaze_headers"] = sorted(self.default_headers)
        return params

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> Any:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = _PROVIDER
        return params

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        messages = self._convert_input(input_).to_messages()
        patched = [
            m.model_copy(update={"content": _rewrite_video_blocks(m.content)})
            if isinstance(m.content, list)
            else m
            for m in messages
        ]
        payload = super()._get_request_payload(patched, stop=stop, **kwargs)
        reasoning = payload.pop("reasoning", None)
        if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
            payload.setdefault("reasoning_effort", reasoning["effort"])
        return payload

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response
            if isinstance(response, dict)
            else response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}}, warnings=False
            )
        )
        side = _extract_side_fields(response_dict)
        for generation in result.generations:
            message = generation.message
            if isinstance(message, AIMessage):
                message.response_metadata["model_provider"] = _PROVIDER
                _apply_side_fields(message, side)
                _strip_tags(message)
                # The streaming path keeps gen.text in step with the stripped content;
                # without this, callbacks and the serialized cache carry the raw tags.
                if isinstance(message.content, str):
                    generation.text = message.content
        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type,
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return generation_chunk
        message = generation_chunk.message
        if isinstance(message, AIMessage):
            message.response_metadata["model_provider"] = _PROVIDER
            side = _extract_side_fields(chunk)
            if side:
                _apply_side_fields(message, side)
        return generation_chunk

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        filt = SideChannelFilter()
        raw: list[str] = []
        emitted: list[str] = []
        seen: set[str] = set()
        for gen in super()._stream(messages, stop=stop, run_manager=None, **kwargs):
            _filter_stream_chunk(gen, filt, raw, emitted)
            _dedupe_side_fields(gen.message, seen)
            if run_manager:
                run_manager.on_llm_new_token(
                    gen.text, chunk=gen, logprobs=(gen.generation_info or {}).get("logprobs")
                )
            yield gen
        final = _final_side_chunk(filt, raw, seen, emitted)
        if final is not None:
            if run_manager:
                run_manager.on_llm_new_token(final.text, chunk=final)
            yield final

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        filt = SideChannelFilter()
        raw: list[str] = []
        emitted: list[str] = []
        seen: set[str] = set()
        async for gen in super()._astream(messages, stop=stop, run_manager=None, **kwargs):
            _filter_stream_chunk(gen, filt, raw, emitted)
            _dedupe_side_fields(gen.message, seen)
            if run_manager:
                await run_manager.on_llm_new_token(
                    gen.text, chunk=gen, logprobs=(gen.generation_info or {}).get("logprobs")
                )
            yield gen
        final = _final_side_chunk(filt, raw, seen, emitted)
        if final is not None:
            if run_manager:
                await run_manager.on_llm_new_token(final.text, chunk=final)
            yield final


__all__ = ["ChatInterfaze"]
