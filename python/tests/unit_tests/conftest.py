from __future__ import annotations

import functools
import json
import operator
from typing import Any

import httpx
import respx

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


def chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
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
INLINE_TAGS = completion(
    "<think>Rayleigh scattering.</think>"
    '<precontext>[{"name": "ocr", "result": {"x": 1}}]</precontext>'
    "The sky is blue."
)
STREAM_CHUNKS: list[dict[str, Any]] = [
    chunk({"content": '<precontext>[{"name":"ocr","result":{"extracted_text":"x"}}]</precontext>'}),
    chunk({"content": "Total "}),
    chunk({"content": "is $12.34"}),
    chunk({}, finish_reason="stop"),
]
THINK_SPLIT: list[dict[str, Any]] = [
    chunk({"content": "<th"}),
    chunk({"content": "ink>Rayleigh scat"}),
    chunk({"content": "tering.</think>The sky "}),
    chunk({"content": "is blue."}),
    chunk({}, finish_reason="stop"),
]
PLAIN_STREAM: list[dict[str, Any]] = [
    chunk({"content": "Hello "}),
    chunk({"content": "world"}),
    chunk({}, finish_reason="stop"),
]
# The same side field on consecutive chunks: `reasoning` would string-concatenate and
# `precontext` would append on merge; `vcache` merges cleanly.
REPEATED_SIDE: list[dict[str, Any]] = [
    chunk({"content": "a"}) | {"reasoning": "why", "precontext": [{"name": "ocr"}], "vcache": True},
    chunk({"content": "b"}) | {"reasoning": "why", "precontext": [{"name": "ocr"}], "vcache": True},
    chunk({}, finish_reason="stop"),
]


def merge(chunks: list[Any]) -> Any:
    return functools.reduce(operator.add, chunks)
