from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

from interfaze import (
    INTERFAZE_BASE_URL,
    INTERFAZE_MODEL,
    InterfazeError,
    SideChannelFilter,
    strip_side_channels,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import Field, SecretStr

_SIDE_FIELDS = ("precontext", "reasoning", "vcache")


def _extract_side_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {k: data[k] for k in _SIDE_FIELDS if data.get(k) is not None}


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
    if reasoning:
        message.response_metadata.setdefault("reasoning", reasoning)
        message.additional_kwargs.setdefault("reasoning", reasoning)
    if precontext:
        message.response_metadata.setdefault("precontext", precontext)
        message.additional_kwargs.setdefault("precontext", precontext)


def _convert_video_block(block: dict[str, Any]) -> dict[str, Any]:
    mime = block.get("mime_type")
    if "url" in block:
        file: dict[str, Any] = {"file_data": block["url"]}
    elif "base64" in block:
        mime = mime or "video/mp4"
        file = {"file_data": f"data:{mime};base64,{block['base64']}"}
    elif "file_id" in block:
        file = {"file_id": block["file_id"]}
    else:
        raise InterfazeError("Video content block requires one of 'url', 'base64', or 'file_id'.")
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


def _filter_stream_chunk(gen: ChatGenerationChunk, filt: SideChannelFilter, raw: list[str]) -> None:
    message = gen.message
    if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content:
        raw.append(message.content)
        message.content = filt.feed(message.content)


def _final_side_chunk(filt: SideChannelFilter, raw: list[str]) -> ChatGenerationChunk | None:
    tail = filt.flush()
    _, reasoning, precontext = strip_side_channels("".join(raw))
    if not tail and not reasoning and not precontext:
        return None
    message = AIMessageChunk(content=tail)
    side: dict[str, Any] = {}
    if reasoning:
        side["reasoning"] = reasoning
    if precontext:
        side["precontext"] = precontext
    _apply_side_fields(message, side)
    return ChatGenerationChunk(message=message)


class ChatInterfaze(ChatOpenAI):
    precontext: list[dict[str, Any]] | None = Field(default=None)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        key = api_key or os.environ.get("INTERFAZE_API_KEY")
        if not key:
            raise InterfazeError(
                "Missing API key. Pass ChatInterfaze(api_key=...) or set the INTERFAZE_API_KEY "
                "environment variable."
            )
        super().__init__(
            api_key=SecretStr(key),
            base_url=base_url or INTERFAZE_BASE_URL,
            model=model or INTERFAZE_MODEL,
            **kwargs,
        )

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
        if self.precontext is not None:
            extra_body = dict(payload.get("extra_body") or {})
            extra_body.setdefault("precontext", self.precontext)
            payload["extra_body"] = extra_body
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
                _apply_side_fields(message, side)
                _strip_tags(message)
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
            side = _extract_side_fields(chunk)
            if side:
                _apply_side_fields(message, side)
        return generation_chunk

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        filt = SideChannelFilter()
        raw: list[str] = []
        for gen in super()._stream(*args, **kwargs):
            _filter_stream_chunk(gen, filt, raw)
            yield gen
        final = _final_side_chunk(filt, raw)
        if final is not None:
            yield final

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        filt = SideChannelFilter()
        raw: list[str] = []
        async for gen in super()._astream(*args, **kwargs):
            _filter_stream_chunk(gen, filt, raw)
            yield gen
        final = _final_side_chunk(filt, raw)
        if final is not None:
            yield final


__all__ = ["ChatInterfaze"]
