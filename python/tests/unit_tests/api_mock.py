"""A catch-all mock of the Interfaze API and the call paths that reach it."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest
import respx
from langchain_core.messages import HumanMessage

from interfaze_langchain import ChatInterfaze
from tests.unit_tests.conftest import BASIC, _sse_bytes, chunk

API_HOST = "api.interfaze.ai"
SCHEME = "Bearer"

# Shapes of the real API errors.
INVALID_KEY_BODY = {
    "error": {
        "message": "The API key provided is not valid",
        "type": "permission_error",
        "code": "insufficient_quota",
        "request_id": "req-test",
    }
}
SIGNUP_HINT = (
    "Get a free API key at https://interfaze.ai and send it as `Authorization: Bearer YOUR_API_KEY`."
)
DAILY_LIMIT_BODY = {
    "error": {
        "message": f"Keyless daily limit of 1000 requests reached. {SIGNUP_HINT}",
        "type": "rate_limit_error",
        "code": "keyless_daily_limit_exceeded",
        "request_id": "req-test",
    }
}
DAILY_LIMIT_HEADERS = {
    "Retry-After": "3600",
    "X-RateLimit-Limit": "1000",
    "X-RateLimit-Remaining": "0",
    "X-RateLimit-Reset": "1900000000",
}
INFLIGHT_BODY = {
    "error": {
        "message": f"Too many keyless requests in progress (limit 5). {SIGNUP_HINT}",
        "type": "rate_limit_error",
        "code": "keyless_inflight_limit_exceeded",
        "request_id": "req-test",
    }
}
INFLIGHT_HEADERS = {"Retry-After": "5"}


def bearer(token: str) -> str:
    return f"{SCHEME} {token}"


_MODELS = {"object": "list", "data": []}
_STREAM = [chunk({"content": "Hi"}), chunk({}, finish_reason="stop")]


def _answer(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/chat/completions") and json.loads(request.content).get("stream"):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_bytes(_STREAM))
    body = BASIC if path.endswith("/chat/completions") else _MODELS
    return httpx.Response(200, json=body)


def mock_api() -> respx.Route:
    """Answer every request to the API host, whatever the path."""
    return respx.route(host=API_HOST).mock(side_effect=_answer)


def mock_status(status: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> respx.Route:
    return respx.route(host=API_HOST).mock(return_value=httpx.Response(status, json=body, headers=headers))


def sent_authorization(route: respx.Route) -> list[list[str]]:
    """The Authorization values of each request, one list per request."""
    return [call.request.headers.get_list("authorization") for call in route.calls]


MESSAGES = [HumanMessage("hi")]
Call = Callable[[ChatInterfaze], Awaitable[object]]


async def _invoke(model: ChatInterfaze) -> object:
    return model.invoke(MESSAGES)


async def _ainvoke(model: ChatInterfaze) -> object:
    return await model.ainvoke(MESSAGES)


async def _stream(model: ChatInterfaze) -> object:
    return list(model.stream(MESSAGES))


async def _astream(model: ChatInterfaze) -> object:
    return [chunk async for chunk in model.astream(MESSAGES)]


async def _models_list(model: ChatInterfaze) -> object:
    return model.root_client.models.list()


async def _async_models_list(model: ChatInterfaze) -> object:
    return await model.root_async_client.models.list()


async def _with_options(model: ChatInterfaze) -> object:
    return model.root_client.with_options(timeout=30).models.list()


async def _async_with_options(model: ChatInterfaze) -> object:
    return await model.root_async_client.with_options(timeout=30).models.list()


ALL_PATHS: dict[str, Call] = {
    "invoke": _invoke,
    "ainvoke": _ainvoke,
    "stream": _stream,
    "astream": _astream,
    "models.list": _models_list,
    "async models.list": _async_models_list,
    "with_options": _with_options,
    "async with_options": _async_with_options,
}


def path_params(*names: str) -> pytest.MarkDecorator:
    """Run a test on the named call paths."""
    return pytest.mark.parametrize("path", [ALL_PATHS[name] for name in names], ids=names)
