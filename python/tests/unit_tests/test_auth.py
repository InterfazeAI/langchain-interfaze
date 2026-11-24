from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import pytest
import respx
from interfaze import AuthenticationError, RateLimitError
from langchain_core.messages import HumanMessage

from interfaze_langchain import ChatInterfaze
from tests.unit_tests.conftest import BASIC, CHAT_URL, PLAIN_STREAM, mock_json, mock_sse

ENV_KEY = "sk-env-synthetic"
EXPLICIT_KEY = "sk-explicit-synthetic"
OPENAI_KEY = "sk-openai-synthetic"

# Error bodies in the shape the API sends: a key it rejects, and a keyless caller over a daily limit.
INVALID_KEY_BODY = {
    "error": {
        "message": "The API key provided is not valid or has been revoked",
        "type": "permission_error",
        "code": "insufficient_quota",
        "request_id": "req-test",
    }
}
KEYLESS_HINT = "Get a free API key at https://interfaze.ai"
KEYLESS_LIMIT_BODY = {
    "error": {
        "message": f"Keyless daily limit of 50 requests reached. {KEYLESS_HINT} and send it as "
        "`Authorization: Bearer <key>`.",
        "type": "rate_limit_error",
        "code": "keyless_daily_limit_exceeded",
        "request_id": "req-test",
    }
}
KEYLESS_LIMIT_HEADERS = {"retry-after": "3600", "x-ratelimit-limit": "50", "x-ratelimit-remaining": "0"}


@pytest.fixture(autouse=True)
def _no_env_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERFAZE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


async def _invoke(model: ChatInterfaze) -> None:
    model.invoke([HumanMessage("hi")])


async def _ainvoke(model: ChatInterfaze) -> None:
    await model.ainvoke([HumanMessage("hi")])


async def _stream(model: ChatInterfaze) -> None:
    list(model.stream([HumanMessage("hi")]))


async def _astream(model: ChatInterfaze) -> None:
    [chunk async for chunk in model.astream([HumanMessage("hi")])]


Call = Callable[[ChatInterfaze], Awaitable[None]]

CALLS = [
    pytest.param(lambda: mock_json(BASIC), _invoke, id="invoke"),
    pytest.param(lambda: mock_json(BASIC), _ainvoke, id="ainvoke"),
    pytest.param(lambda: mock_sse(PLAIN_STREAM), _stream, id="stream"),
    pytest.param(lambda: mock_sse(PLAIN_STREAM), _astream, id="astream"),
]

ERROR_CALLS = [
    pytest.param(call, id=call.__name__.lstrip("_")) for call in (_invoke, _ainvoke, _stream, _astream)
]


async def _sent_headers(mock: Callable[[], respx.Route], call: Call, model: ChatInterfaze) -> httpx.Headers:
    route = mock()
    await call(model)
    return route.calls.last.request.headers


def _mock_error(status: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> respx.Route:
    return respx.post(CHAT_URL).mock(return_value=httpx.Response(status, json=body, headers=headers))


@respx.mock
@pytest.mark.parametrize(("mock", "call"), CALLS)
async def test_no_key_sends_no_authorization(mock: Callable[[], respx.Route], call: Call) -> None:
    assert "authorization" not in await _sent_headers(mock, call, ChatInterfaze())


@respx.mock
@pytest.mark.parametrize(("mock", "call"), CALLS)
async def test_no_key_ignores_openai_api_key(
    monkeypatch: pytest.MonkeyPatch, mock: Callable[[], respx.Route], call: Call
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    headers = await _sent_headers(mock, call, ChatInterfaze())
    assert "authorization" not in headers
    assert not any(OPENAI_KEY in value for value in headers.values())


@respx.mock
def test_no_key_keeps_caller_request_headers() -> None:
    route = mock_json(BASIC)
    ChatInterfaze().invoke([HumanMessage("hi")], extra_headers={"x-trace": "1"})
    headers = route.calls.last.request.headers
    assert headers["x-trace"] == "1"
    assert "authorization" not in headers


@respx.mock
@pytest.mark.parametrize(("mock", "call"), CALLS)
async def test_env_key_authorizes(
    monkeypatch: pytest.MonkeyPatch, mock: Callable[[], respx.Route], call: Call
) -> None:
    monkeypatch.setenv("INTERFAZE_API_KEY", ENV_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    headers = await _sent_headers(mock, call, ChatInterfaze())
    assert headers["authorization"] == f"Bearer {ENV_KEY}"


@respx.mock
@pytest.mark.parametrize(("mock", "call"), CALLS)
async def test_explicit_key_beats_env_key(
    monkeypatch: pytest.MonkeyPatch, mock: Callable[[], respx.Route], call: Call
) -> None:
    monkeypatch.setenv("INTERFAZE_API_KEY", ENV_KEY)
    headers = await _sent_headers(mock, call, ChatInterfaze(api_key=EXPLICIT_KEY))
    assert headers["authorization"] == f"Bearer {EXPLICIT_KEY}"


@respx.mock
@pytest.mark.parametrize("call", ERROR_CALLS)
async def test_rejected_key_raises_401_without_a_keyless_retry(call: Call) -> None:
    route = _mock_error(401, INVALID_KEY_BODY)
    with pytest.raises(AuthenticationError) as caught:
        await call(ChatInterfaze(api_key=EXPLICIT_KEY, max_retries=2))
    assert caught.value.status_code == 401
    assert caught.value.code == "insufficient_quota"
    assert [c.request.headers.get("authorization") for c in route.calls] == [f"Bearer {EXPLICIT_KEY}"]


@respx.mock
@pytest.mark.parametrize("call", ERROR_CALLS)
async def test_keyless_limit_raises_429_with_the_key_hint(call: Call) -> None:
    route = _mock_error(429, KEYLESS_LIMIT_BODY, KEYLESS_LIMIT_HEADERS)
    with pytest.raises(RateLimitError) as caught:
        await call(ChatInterfaze(max_retries=0))
    assert caught.value.status_code == 429
    assert caught.value.code == "keyless_daily_limit_exceeded"
    assert KEYLESS_HINT in caught.value.message
    assert caught.value.response.headers["retry-after"] == "3600"
    assert "authorization" not in route.calls.last.request.headers
