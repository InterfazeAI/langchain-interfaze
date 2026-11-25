from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from typing import Any, NamedTuple

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
from openai import AsyncOpenAI, OpenAI
from pydantic import SecretStr, model_validator
from typing_extensions import Self

from interfaze_langchain._version import __version__

_PROVIDER = "interfaze"

_DEFAULT_TIMEOUT = 900.0

_HEADER_SHOW_ADDITIONAL_INFO = "x-show-additional-info"
_HEADER_BYPASS_MOA = "x-interfaze-bypass-moa"
_HEADER_BYPASS_CACHE = "x-interfaze-bypass-cache"

_ENV_API_KEY = "INTERFAZE_API_KEY"

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


def _default_role(response: Any, part_key: str) -> None:
    """Default a missing `role` to assistant, in place.

    Interfaze sends `role` on the first delta only. Without it the parent builds a
    ChatMessage, which pydantic rejects for role=None and which carries no
    additional_kwargs. Accepts dict and pydantic shapes; the two paths differ.
    """
    choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    for choice in choices or ():
        part = choice.get(part_key) if isinstance(choice, dict) else getattr(choice, part_key, None)
        if isinstance(part, dict):
            if not part.get("role"):
                part["role"] = "assistant"
        elif part is not None and not getattr(part, "role", None):
            part.role = "assistant"


_ResolvedKey = SecretStr | Callable[[], str] | Callable[[], Awaitable[str]]
_ApiKey = str | _ResolvedKey | None


def _keyless_api_key() -> str:
    """Stands in for the key on a model with no Interfaze key.

    The OpenAI SDK rejects an empty string key but accepts a callable, and it builds no
    Authorization header from the empty string the callable returns.
    """
    return ""


# langchain-openai wraps the sync stand-in for the async client, so that client gets its own.
async def _keyless_async_api_key() -> str:
    return ""


def _trim_key(key: Any) -> Any:
    """Trim a string key; a blank key becomes empty. A callable passes through untouched."""
    text = key.get_secret_value() if isinstance(key, SecretStr) else key
    return text.strip() if isinstance(text, str) else text


def _resolve_key(*explicit: _ApiKey) -> _ResolvedKey | None:
    """The first usable key: an explicit one, then INTERFAZE_API_KEY, else None (keyless)."""
    candidates = (*(_trim_key(key) for key in explicit), _trim_key(os.environ.get(_ENV_API_KEY)))
    key = next((candidate for candidate in candidates if candidate), None)
    return SecretStr(key) if isinstance(key, str) else key


# openai builds the User-Agent header from the class name, so the names omit the underscore.
# The two clients below replace the plain OpenAI clients that ChatOpenAI builds, so every
# path through `root_client`, `client`, `async_client`, `with_options` and `copy` shares them.
class InterfazeOpenAI(OpenAI):
    # openai reads OPENAI_ADMIN_KEY for its admin resources. No admin key may reach Interfaze.
    @property
    def _admin_api_key_auth(self) -> dict[str, str]:
        return {}

    # openai-python refuses a request with no Authorization header. Interfaze accepts it
    # from a keyless client only.
    def _validate_headers(self, *args: Any, **kwargs: Any) -> None:
        if self._api_key_provider is not _keyless_api_key:
            super()._validate_headers(*args, **kwargs)


class AsyncInterfazeOpenAI(AsyncOpenAI):
    @property
    def _admin_api_key_auth(self) -> dict[str, str]:
        return {}

    def _validate_headers(self, *args: Any, **kwargs: Any) -> None:
        if self._api_key_provider is not _keyless_async_api_key:
            super()._validate_headers(*args, **kwargs)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


# One id for each key function, so the cache key tells two functions apart without exposing either.
_KEY_FUNCTION_IDS: weakref.WeakKeyDictionary[Callable[..., Any], int] = weakref.WeakKeyDictionary()
_KEY_FUNCTION_COUNTER = itertools.count(1)


def _key_function_id(function: Callable[..., Any]) -> int:
    return _KEY_FUNCTION_IDS.setdefault(function, next(_KEY_FUNCTION_COUNTER))


def _redact_headers(headers: Mapping[str, str]) -> list[str]:
    """Fingerprint caller header values; show only the flags this package owns.

    `_identifying_params` reaches the LLM cache key and the `invocation_params` LangSmith
    records, and distinctness is all the key needs.
    """
    public = (_HEADER_SHOW_ADDITIONAL_INFO, _HEADER_BYPASS_MOA, _HEADER_BYPASS_CACHE)
    return [f"{k}={headers[k]}" if k in public else f"{k}#{_digest(headers[k])}" for k in sorted(headers)]


def _apply_side_fields(message: AIMessage, side: dict[str, Any]) -> None:
    for key, value in side.items():
        message.response_metadata[key] = value
        message.additional_kwargs[key] = value


def _strip_tags(message: AIMessage, truncated: bool = False) -> None:
    if not isinstance(message.content, str) or not (
        "<think>" in message.content or "<precontext>" in message.content
    ):
        return
    text, reasoning, precontext = strip_side_channels(message.content)
    if truncated:
        recovered = _recover_tail(message.content, emitted="", truncated=True)
        text = recovered.tail.strip()
        reasoning = reasoning or recovered.reasoning
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
    rewritten = []
    for block in content:
        if not isinstance(block, dict):
            rewritten.append(block)
        elif block.get("type") == "video":
            rewritten.append(_convert_video_block(block))
        elif block.get("file_id") is not None or (
            isinstance(block.get("file"), dict) and block["file"].get("file_id") is not None
        ):
            # Interfaze has no file store, so a file_id reference can only 400 downstream.
            # Both the standard block shape and the openai-native nesting under `file`.
            raise InterfazeError(
                "Interfaze cannot resolve content by 'file_id'. Pass 'url' or 'base64' instead."
            )
        else:
            rewritten.append(block)
    return rewritten if rewritten != content else content


def _fingerprint(key: str, value: Any) -> str:
    """`precontext`/`reasoning` accumulate, so only an identical payload is a duplicate;
    `vcache` is scalar state, deduped by name because bool merges as int."""
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


class _SideChannelStream:
    """Per-stream bookkeeping for the side-channel filter.

    `_stream` and `_astream` are otherwise identical, so holding the state here leaves the
    two loops differing only by `async`.
    """

    def __init__(self) -> None:
        self._filter = SideChannelFilter()
        self._raw: list[str] = []
        self._emitted: list[str] = []
        self._seen: set[str] = set()
        self._finish_reason: str | None = None

    def absorb(self, gen: ChatGenerationChunk) -> None:
        message = gen.message
        if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content:
            self._raw.append(message.content)
            message.content = self._filter.feed(message.content)
            self._emitted.append(message.content)
            gen.text = message.content
        _dedupe_side_fields(gen.message, self._seen)
        self._finish_reason = (gen.generation_info or {}).get("finish_reason") or self._finish_reason

    def final_chunk(self) -> ChatGenerationChunk | None:
        tail = self._filter.flush()
        joined = "".join(self._raw)
        _, reasoning, precontext = strip_side_channels(joined)
        if not tail:
            recovered = _recover_tail(joined, "".join(self._emitted), self._finish_reason == "length")
            tail = recovered.tail
            reasoning = reasoning or recovered.reasoning
        side: dict[str, Any] = {}
        if reasoning and _fingerprint("reasoning", reasoning) not in self._seen:
            side["reasoning"] = reasoning
        if precontext and _fingerprint("precontext", precontext) not in self._seen:
            side["precontext"] = precontext
        if not tail and not side:
            return None
        message = AIMessageChunk(content=tail)
        message.response_metadata["model_provider"] = _PROVIDER
        _apply_side_fields(message, side)
        return ChatGenerationChunk(message=message)


def _first_effort(*sources: Any) -> Any:
    for source in sources:
        effort = source.get("effort") if isinstance(source, dict) else source
        if effort is not None:
            return effort
    return None


def _without_closed_blocks(raw: str) -> str:
    """Closed blocks removed, no trim — `strip_side_channels` trims and breaks prefix compares."""
    return re.sub(r"<precontext>[\s\S]*?</precontext>", "", re.sub(r"<think>[\s\S]*?</think>", "", raw))


class _OpenTag(NamedTuple):
    tag: str
    before: str
    after: str


class _Recovered(NamedTuple):
    tail: str
    reasoning: str | None


def _open_side_channel(text: str) -> _OpenTag | None:
    """The earliest unmatched opening tag.

    Earliest by position, not tag order: a truncated answer whose prose mentions
    `<think>` before an unclosed `<precontext>` must split at the precontext.
    """
    positions = [(text.find(f"<{tag}>"), tag) for tag in ("think", "precontext")]
    present = [(at, tag) for at, tag in positions if at != -1]
    if not present:
        return None
    at, tag = min(present)
    return _OpenTag(tag, text[:at], text[at + len(tag) + 2 :])


def _recover_tail(raw: str, emitted: str, truncated: bool) -> _Recovered:
    """What the caller still owes, given what already streamed.

    An unmatched tag is prose in a completed response and an unclosed side channel in a
    truncated one, so `truncated` decides. A partial `<think>` becomes reasoning; a
    partial `<precontext>` is unparseable and dropped.
    """
    text = _without_closed_blocks(raw)
    open_tag = _open_side_channel(text) if truncated else None
    visible = open_tag.before if open_tag else text
    tail = visible[len(emitted) :] if visible.startswith(emitted) else ""
    if open_tag and open_tag.tag == "think" and open_tag.after:
        return _Recovered(tail, open_tag.after)
    return _Recovered(tail, None)


class ChatInterfaze(ChatOpenAI):
    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        return ["interfaze_langchain", "chat_models"]

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"openai_api_key": "INTERFAZE_API_KEY"}

    # Provider family, not the model: `interfaze-beta` reaches tracing and the cache key
    # via `ls_model_name` / `model_name`. Mirrors ChatOpenAI's "openai-chat".
    @property
    def _llm_type(self) -> str:
        return "interfaze"

    def __init__(
        self,
        *,
        api_key: _ApiKey = None,
        base_url: str | None = None,
        model: str | None = None,
        show_additional_info: bool = False,
        bypass_moa: bool = False,
        bypass_cache: bool = False,
        default_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """The three flags map to Interfaze control headers. `show_additional_info` is the
        only way to get precontext while streaming; non-streaming responses always carry
        it. `bypass_cache` matters when you need `reasoning`, which a cache hit omits.

        The key is `api_key`, then `openai_api_key`, then `INTERFAZE_API_KEY`. A blank string
        is not a key. Without a key, requests carry no Authorization header and use
        Interfaze's keyless access. `OPENAI_API_KEY` is never read.
        """
        key = _resolve_key(api_key, kwargs.pop("openai_api_key", None))
        headers = {k.lower(): str(v) for k, v in (default_headers or {}).items()}
        if not headers.get("authorization", "").strip():
            headers.pop("authorization", None)
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
            api_key=key or _keyless_api_key,
            base_url=base_url or INTERFAZE_BASE_URL,
            model=model or INTERFAZE_MODEL,
            default_headers=headers or None,
            **kwargs,
        )
        # The parent builds plain OpenAI clients; a client the caller passes in stays as given.
        if kwargs.get("root_client") is None and type(self.root_client) is OpenAI:
            self.root_client.__class__ = InterfazeOpenAI
        if kwargs.get("root_async_client") is None and type(self.root_async_client) is AsyncOpenAI:
            self.root_async_client.__class__ = AsyncInterfazeOpenAI
            if key is None:
                self.root_async_client._api_key_provider = _keyless_async_api_key
        # The stand-in key function stays in the built clients; the field holds a plain blank key.
        if key is None:
            self.openai_api_key = SecretStr("")

    # Must be uniquely named: pydantic replaces same-named validators rather than chaining
    # them, so reusing the parent's name would drop its version entry.
    @model_validator(mode="after")
    def _set_interfaze_version(self) -> Self:
        self._add_version("interfaze-langchain", __version__)
        return self

    @property
    def _identifying_params(self) -> dict[str, Any]:
        # Without these, set_llm_cache serves a bypass_cache model the plain model's answer,
        # and one tenant's key the answer cached under another's.
        params = {**super()._identifying_params, "_type": self._llm_type}
        # A key function resolves per request, so its id stands for it.
        if isinstance(self.openai_api_key, SecretStr):
            params["interfaze_key"] = _digest(self.openai_api_key.get_secret_value())
        elif callable(self.openai_api_key):
            params["interfaze_key"] = f"function-{_key_function_id(self.openai_api_key)}"
        if self.default_headers:
            params["interfaze_headers"] = _redact_headers(self.default_headers)
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
        # Interfaze has no `reasoning` param; fold it into reasoning_effort. A per-call
        # value always beats a model-level one.
        payload.pop("reasoning", None)
        effort = _first_effort(
            kwargs.get("reasoning"), kwargs.get("reasoning_effort"), self.reasoning, self.reasoning_effort
        )
        if effort is not None:
            payload["reasoning_effort"] = effort
        return payload

    def _create_chat_result(
        self,
        response: Any,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        _default_role(response, "message")
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
                _strip_tags(message, (generation.generation_info or {}).get("finish_reason") == "length")
                # gen.text feeds callbacks and the serialized cache, so keep it in step
                # with the stripped content or both carry the raw tags.
                if isinstance(message.content, str):
                    generation.text = message.content
        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type,
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        # `with_structured_output` streams through beta.chat.completions, which nests the
        # frame under "chunk" — side fields ride the envelope, so unwrap before reading.
        body = chunk.get("chunk") or chunk
        _default_role(body, "delta")
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is None:
            return generation_chunk
        message = generation_chunk.message
        if isinstance(message, AIMessage):
            message.response_metadata["model_provider"] = _PROVIDER
            side = _extract_side_fields(body)
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
        stream = _SideChannelStream()
        # run_manager is withheld deliberately: ChatOpenAI fires on_llm_new_token before
        # yielding, so handlers would see unfiltered `<think>`. Fired below instead.
        for gen in super()._stream(messages, stop=stop, run_manager=None, **kwargs):
            stream.absorb(gen)
            if run_manager:
                run_manager.on_llm_new_token(
                    gen.text, chunk=gen, logprobs=(gen.generation_info or {}).get("logprobs")
                )
            yield gen
        final = stream.final_chunk()
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
        stream = _SideChannelStream()
        # See _stream: the manager is withheld so handlers never see unfiltered `<think>`.
        async for gen in super()._astream(messages, stop=stop, run_manager=None, **kwargs):
            stream.absorb(gen)
            if run_manager:
                await run_manager.on_llm_new_token(
                    gen.text, chunk=gen, logprobs=(gen.generation_info or {}).get("logprobs")
                )
            yield gen
        final = stream.final_chunk()
        if final is not None:
            if run_manager:
                await run_manager.on_llm_new_token(final.text, chunk=final)
            yield final


__all__ = ["ChatInterfaze"]
