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

# Interfaze runs OCR / web search / scraping / STT / forecasting inline, so a single
# completion can legitimately take minutes. Matches the core `interfaze` SDK default.
_DEFAULT_TIMEOUT = 900.0

# Interfaze control-plane headers (mirrors `interfaze._constants`).
_HEADER_SHOW_ADDITIONAL_INFO = "x-show-additional-info"
_HEADER_BYPASS_MOA = "x-interfaze-bypass-moa"
_HEADER_BYPASS_CACHE = "x-interfaze-bypass-cache"
_HEADER_ADMIN_KEY = "x-admin-key"

_SIDE_FIELDS = ("precontext", "reasoning", "vcache")

# Video containers Interfaze accepts, mirroring `interfaze.inputs`.
_VIDEO_MIME = {
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "webm": "video/webm",
    "avi": "video/x-msvideo",
    "mkv": "video/x-matroska",
    "3gp": "video/3gpp",
}


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


def _video_mime_from_url(url: str) -> str | None:
    base = url.split("?")[0].split("#")[0]
    ext = base.rsplit(".", 1)[-1].lower() if "." in base else ""
    return _VIDEO_MIME.get(ext)


def _convert_video_block(block: dict[str, Any]) -> dict[str, Any]:
    # Interfaze has no file store: the `file` part accepts `file_data` only.
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


def _filter_stream_chunk(gen: ChatGenerationChunk, filt: SideChannelFilter, raw: list[str]) -> None:
    message = gen.message
    if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content:
        raw.append(message.content)
        message.content = filt.feed(message.content)
        # `gen.text` was snapshotted from the unfiltered content at construction, and it
        # is what feeds on_llm_new_token / the event bridge — keep it in sync.
        gen.text = message.content


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
    """Interfaze chat model.

    Wraps the Interfaze `/v1/chat/completions` endpoint and surfaces the extra fields
    Interfaze returns — `precontext`, `reasoning`, `vcache` — on both
    `response_metadata` and `additional_kwargs`.
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain_interfaze", "chat_models"]

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "INTERFAZE_API_KEY"}

    @property
    def _llm_type(self) -> str:
        return "interfaze-chat"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        show_additional_info: bool = False,
        bypass_moa: bool = False,
        bypass_cache: bool = False,
        admin_key: str | None = None,
        default_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Build an Interfaze chat model.

        Args:
            api_key: Interfaze API key; falls back to `INTERFAZE_API_KEY`.
            base_url: Overrides the Interfaze endpoint.
            model: Defaults to `interfaze-beta`.
            show_additional_info: Emit inline `<precontext>` blocks while streaming.
                Interfaze only sends streamed precontext when this is on.
            bypass_moa: Skip the mixture-of-architecture internal tool router.
            bypass_cache: Skip the semantic cache.
            admin_key: Admin key that surfaces a `debug` field.
            default_headers: Extra headers merged with the Interfaze control headers.
            kwargs: Forwarded to `ChatOpenAI`.
        """
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
        if admin_key:
            headers[_HEADER_ADMIN_KEY] = admin_key
        if "timeout" not in kwargs and "request_timeout" not in kwargs:
            kwargs["timeout"] = _DEFAULT_TIMEOUT
        # langchain-openai only auto-enables `stream_options.include_usage` for OpenAI's
        # own base URL, so a custom endpoint silently loses `usage_metadata` on every
        # streamed response. Interfaze supports it; match the JS package, which defaults on.
        kwargs.setdefault("stream_usage", True)
        # Interfaze speaks Chat Completions only; never let a stray `reasoning=` kwarg or
        # LC_OUTPUT_VERSION reroute the request to the OpenAI Responses API, which would
        # bypass every hook below.
        kwargs["use_responses_api"] = False
        super().__init__(
            api_key=SecretStr(key),
            base_url=base_url or INTERFAZE_BASE_URL,
            model=model or INTERFAZE_MODEL,
            default_headers=headers or None,
            **kwargs,
        )

    # Must be uniquely named: pydantic replaces same-named validators rather than
    # chaining them, so reusing the parent's name would drop its version entry.
    @model_validator(mode="after")
    def _set_interfaze_version(self) -> Self:
        self._add_version("langchain-interfaze", __version__)
        return self

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
        return super()._get_request_payload(patched, stop=stop, **kwargs)

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
        # `run_manager` is deliberately withheld from super(): ChatOpenAI fires
        # on_llm_new_token *before* yielding, i.e. before this filter runs, so token
        # handlers would see raw `<think>`/`<precontext>` text. Core's own stream()
        # doesn't pass a manager down, but the v2 protocol path does. Fire it here
        # instead, once the chunk is clean.
        for gen in super()._stream(messages, stop=stop, run_manager=None, **kwargs):
            _filter_stream_chunk(gen, filt, raw)
            if run_manager:
                run_manager.on_llm_new_token(
                    gen.text, chunk=gen, logprobs=(gen.generation_info or {}).get("logprobs")
                )
            yield gen
        final = _final_side_chunk(filt, raw)
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
        async for gen in super()._astream(messages, stop=stop, run_manager=None, **kwargs):
            _filter_stream_chunk(gen, filt, raw)
            if run_manager:
                await run_manager.on_llm_new_token(
                    gen.text, chunk=gen, logprobs=(gen.generation_info or {}).get("logprobs")
                )
            yield gen
        final = _final_side_chunk(filt, raw)
        if final is not None:
            if run_manager:
                await run_manager.on_llm_new_token(final.text, chunk=final)
            yield final


__all__ = ["ChatInterfaze"]
