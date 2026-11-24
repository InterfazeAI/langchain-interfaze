from __future__ import annotations

import time
from typing import Any

import httpx
import openai
import pytest
import respx
from openai._base_client import AsyncAPIClient, SyncAPIClient

from interfaze_langchain import ChatInterfaze
from tests.unit_tests.api_mock import (
    DAILY_LIMIT_BODY,
    DAILY_LIMIT_HEADERS,
    INFLIGHT_BODY,
    INFLIGHT_HEADERS,
    INVALID_KEY_BODY,
    MESSAGES,
    SIGNUP_HINT,
    Call,
    mock_status,
)
from tests.unit_tests.conftest import BASIC

DEFAULT_ATTEMPTS = 3  # one request plus the two default retries
FAST_SECONDS = 2.0
EXPLICIT_KEY = "explicit-key-2"
SERVER_ERROR_BODY = {
    "error": {"message": "boom", "type": "server_error", "code": "internal", "request_id": "r"}
}
RATE_LIMIT_BODY = {
    "error": {"message": "slow down", "type": "rate_limit_error", "code": "rate_limit_exceeded"}
}


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


ERROR_PATHS = pytest.mark.parametrize(
    "path",
    [_invoke, _ainvoke, _stream, _astream, _models_list],
    ids=["invoke", "ainvoke", "stream", "astream", "models.list"],
)
KEY_KWARGS = pytest.mark.parametrize("kwargs", [{}, {"api_key": EXPLICIT_KEY}], ids=["no key", "api_key"])


@pytest.fixture
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the wait between retries so a retry count costs no time."""

    def sync_sleep(self: Any, **kwargs: Any) -> None:
        return None

    async def async_sleep(self: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(SyncAPIClient, "_sleep_for_retry", sync_sleep)
    monkeypatch.setattr(AsyncAPIClient, "_sleep_for_retry", async_sleep)


async def _fails(path: Call, model: ChatInterfaze) -> openai.APIStatusError:
    with pytest.raises(openai.APIStatusError) as caught:
        await path(model)
    return caught.value


@respx.mock
@ERROR_PATHS
@KEY_KWARGS
async def test_401_reaches_the_caller_after_one_request(
    kwargs: dict[str, Any], path: Call, no_retry_sleep: None
) -> None:
    route = mock_status(401, INVALID_KEY_BODY)
    error = await _fails(path, ChatInterfaze(**kwargs))
    assert route.call_count == 1
    assert isinstance(error, openai.AuthenticationError)
    assert error.status_code == 401
    assert error.code == "insufficient_quota"
    assert error.type == "permission_error"
    assert "The API key provided is not valid" in error.message
    assert error.response.json() == INVALID_KEY_BODY


@respx.mock
@ERROR_PATHS
async def test_keyless_401_is_not_retried_with_another_key(
    monkeypatch: pytest.MonkeyPatch, path: Call, no_retry_sleep: None
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-4")
    route = mock_status(401, INVALID_KEY_BODY)
    await _fails(path, ChatInterfaze())
    assert [call.request.headers.get("authorization") for call in route.calls] == [None]


@respx.mock
@ERROR_PATHS
async def test_daily_limit_429_reaches_the_caller_after_one_request_with_no_wait(
    path: Call,
) -> None:
    route = mock_status(429, DAILY_LIMIT_BODY, DAILY_LIMIT_HEADERS)
    started = time.perf_counter()
    error = await _fails(path, ChatInterfaze())
    assert time.perf_counter() - started < FAST_SECONDS
    assert route.call_count == 1
    assert isinstance(error, openai.RateLimitError)
    assert error.status_code == 429
    assert error.code == "keyless_daily_limit_exceeded"
    assert error.type == "rate_limit_error"
    assert SIGNUP_HINT in error.message
    assert error.response.json() == DAILY_LIMIT_BODY
    assert error.response.headers["retry-after"] == DAILY_LIMIT_HEADERS["Retry-After"]
    assert error.response.headers["x-ratelimit-remaining"] == "0"


@respx.mock
@pytest.mark.parametrize(
    "path", [_invoke, _ainvoke, _stream, _models_list], ids=["invoke", "ainvoke", "stream", "models.list"]
)
async def test_daily_limit_429_is_found_by_its_body_code_not_by_retry_after(path: Call) -> None:
    route = mock_status(429, DAILY_LIMIT_BODY, {"Retry-After": "30"})
    error = await _fails(path, ChatInterfaze())
    assert route.call_count == 1
    assert error.code == "keyless_daily_limit_exceeded"


@respx.mock
@ERROR_PATHS
async def test_inflight_429_is_retried_as_before(path: Call, no_retry_sleep: None) -> None:
    route = mock_status(429, INFLIGHT_BODY, INFLIGHT_HEADERS)
    error = await _fails(path, ChatInterfaze())
    assert route.call_count == DEFAULT_ATTEMPTS
    assert error.status_code == 429
    assert error.code == "keyless_inflight_limit_exceeded"
    assert SIGNUP_HINT in error.message
    assert error.response.headers["retry-after"] == "5"


@respx.mock
async def test_inflight_429_that_clears_returns_the_answer(no_retry_sleep: None) -> None:
    route = respx.route(host="api.interfaze.ai").mock(
        side_effect=[
            httpx.Response(429, json=INFLIGHT_BODY, headers=INFLIGHT_HEADERS),
            httpx.Response(200, json=BASIC),
        ]
    )
    assert ChatInterfaze().invoke(MESSAGES).content == "Hi!"
    assert route.call_count == 2


@respx.mock
@KEY_KWARGS
@pytest.mark.parametrize(
    ("status", "body"),
    [(500, SERVER_ERROR_BODY), (503, SERVER_ERROR_BODY), (429, RATE_LIMIT_BODY)],
    ids=["500", "503", "plain 429"],
)
async def test_other_retryable_errors_keep_the_default_retries(
    kwargs: dict[str, Any], status: int, body: dict[str, Any], no_retry_sleep: None
) -> None:
    route = mock_status(status, body)
    await _fails(_invoke, ChatInterfaze(**kwargs))
    assert route.call_count == DEFAULT_ATTEMPTS


@respx.mock
async def test_max_retries_still_bounds_the_attempts(no_retry_sleep: None) -> None:
    route = mock_status(500, SERVER_ERROR_BODY)
    await _fails(_invoke, ChatInterfaze(max_retries=1))
    assert route.call_count == 2


@respx.mock
async def test_daily_limit_with_a_key_follows_the_default_retries(no_retry_sleep: None) -> None:
    route = mock_status(429, DAILY_LIMIT_BODY, {"Retry-After": "30"})
    await _fails(_invoke, ChatInterfaze(api_key=EXPLICIT_KEY))
    assert route.call_count == DEFAULT_ATTEMPTS


@respx.mock
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(429, text="busy"),
        httpx.Response(429, json=["not", "an", "error"]),
        httpx.Response(429, json={"error": "busy"}),
    ],
    ids=["not json", "json list", "error is a string"],
)
async def test_429_with_an_unreadable_error_body_keeps_the_default_retries(
    response: httpx.Response, no_retry_sleep: None
) -> None:
    route = respx.route(host="api.interfaze.ai").mock(return_value=response)
    await _fails(_invoke, ChatInterfaze())
    assert route.call_count == DEFAULT_ATTEMPTS


@respx.mock
@pytest.mark.parametrize("status", [500, 503])
async def test_server_error_with_the_daily_limit_code_keeps_the_default_retries(
    status: int, no_retry_sleep: None
) -> None:
    route = mock_status(status, DAILY_LIMIT_BODY)
    await _fails(_invoke, ChatInterfaze())
    assert route.call_count == DEFAULT_ATTEMPTS
