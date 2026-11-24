"""Tests on a real localhost socket.

`httpx.MockTransport` and respx skip the HTTP wire layer, so they cannot show that a blank
key breaks the connection, or what a streamed async 429 looks like before its body is read.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import openai
import pytest
from openai._base_client import AsyncAPIClient, SyncAPIClient

from interfaze_langchain import ChatInterfaze
from tests.unit_tests.api_mock import (
    DAILY_LIMIT_BODY,
    DAILY_LIMIT_HEADERS,
    INFLIGHT_BODY,
    MESSAGES,
    SIGNUP_HINT,
    bearer,
)
from tests.unit_tests.conftest import BASIC

ENV_KEY = "env-key-1"
EXPLICIT_KEY = "explicit-key-2"
CALLER_AUTH = bearer("caller-token-6")
DEFAULT_ATTEMPTS = 3
FAST_SECONDS = 2.0
SHORT_WAIT = 30
LONG_WAIT = 61
LIMIT_WAIT = 60


class FakeApi:
    """Records the Authorization headers of each request and replies with a fixed answer."""

    def __init__(self) -> None:
        self.authorization: list[list[str]] = []
        self.status = 200
        self.body: dict[str, Any] = BASIC
        self.headers: dict[str, str] = {}
        self.base_url = ""

    def reply(self, status: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> None:
        self.status, self.body, self.headers = status, body, headers or {}


@pytest.fixture
def api(socket_enabled: None) -> Iterator[FakeApi]:
    fake = FakeApi()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return None

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("content-length") or 0))
            fake.authorization.append(self.headers.get_all("authorization") or [])
            payload = json.dumps(fake.body).encode()
            self.send_response(fake.status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            [self.send_header(name, value) for name, value in fake.headers.items()]
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    fake.base_url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield fake
    server.shutdown()
    server.server_close()


@pytest.fixture
def no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    def sync_sleep(self: Any, **kwargs: Any) -> None:
        return None

    async def async_sleep(self: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(SyncAPIClient, "_sleep_for_retry", sync_sleep)
    monkeypatch.setattr(AsyncAPIClient, "_sleep_for_retry", async_sleep)


@pytest.mark.parametrize("key", ["", " ", " \n"])
async def test_blank_explicit_key_sends_no_header_on_the_wire(api: FakeApi, key: str) -> None:
    model = ChatInterfaze(api_key=key, base_url=api.base_url)
    model.invoke(MESSAGES)
    await model.ainvoke(MESSAGES)
    assert api.authorization == [[], []]


@pytest.mark.parametrize("env", ["", " ", " \n"])
async def test_blank_env_key_sends_no_header_on_the_wire(
    monkeypatch: pytest.MonkeyPatch, api: FakeApi, env: str
) -> None:
    monkeypatch.setenv("INTERFAZE_API_KEY", env)
    model = ChatInterfaze(base_url=api.base_url)
    model.invoke(MESSAGES)
    await model.ainvoke(MESSAGES)
    assert api.authorization == [[], []]


@pytest.mark.parametrize("key", ["", " ", " \n"])
async def test_blank_explicit_key_sends_the_env_key_on_the_wire(
    monkeypatch: pytest.MonkeyPatch, api: FakeApi, key: str
) -> None:
    monkeypatch.setenv("INTERFAZE_API_KEY", f" {ENV_KEY}\n")
    model = ChatInterfaze(api_key=key, base_url=api.base_url)
    model.invoke(MESSAGES)
    await model.ainvoke(MESSAGES)
    assert api.authorization == [[bearer(ENV_KEY)]] * 2


async def test_openai_api_key_field_is_sent_on_the_wire(api: FakeApi) -> None:
    ChatInterfaze(openai_api_key=EXPLICIT_KEY, base_url=api.base_url).invoke(MESSAGES)
    assert api.authorization == [[bearer(EXPLICIT_KEY)]]


@pytest.mark.parametrize("kwargs", [{}, {"api_key": EXPLICIT_KEY}], ids=["no key", "api_key"])
async def test_caller_authorization_header_is_sent_once_on_the_wire(
    api: FakeApi, kwargs: dict[str, Any]
) -> None:
    model = ChatInterfaze(base_url=api.base_url, default_headers={"Authorization": CALLER_AUTH}, **kwargs)
    model.invoke(MESSAGES)
    await model.ainvoke(MESSAGES)
    model.invoke(MESSAGES, extra_headers={"AUTHORIZATION": CALLER_AUTH})
    assert api.authorization == [[CALLER_AUTH]] * 3


@pytest.mark.parametrize("wait", [DAILY_LIMIT_HEADERS["Retry-After"], str(LONG_WAIT)])
async def test_streamed_daily_limit_after_one_request_whatever_the_body(api: FakeApi, wait: str) -> None:
    api.reply(429, DAILY_LIMIT_BODY, {**DAILY_LIMIT_HEADERS, "Retry-After": wait})
    model = ChatInterfaze(base_url=api.base_url)
    started = time.perf_counter()
    with pytest.raises(openai.RateLimitError) as sync_caught:
        list(model.stream(MESSAGES))
    with pytest.raises(openai.RateLimitError) as async_caught:
        [chunk async for chunk in model.astream(MESSAGES)]
    assert time.perf_counter() - started < FAST_SECONDS
    assert api.authorization == [[], []]
    assert sync_caught.value.code == async_caught.value.code == "keyless_daily_limit_exceeded"
    assert SIGNUP_HINT in async_caught.value.message


async def test_streamed_async_429_with_a_short_wait_keeps_the_default_retries(
    api: FakeApi, no_retry_sleep: None
) -> None:
    api.reply(429, DAILY_LIMIT_BODY, {"Retry-After": str(LIMIT_WAIT)})
    with pytest.raises(openai.RateLimitError):
        [chunk async for chunk in ChatInterfaze(base_url=api.base_url).astream(MESSAGES)]
    assert len(api.authorization) == DEFAULT_ATTEMPTS


async def test_streamed_inflight_429_keeps_the_default_retries(api: FakeApi, no_retry_sleep: None) -> None:
    api.reply(429, INFLIGHT_BODY, {"Retry-After": "5"})
    model = ChatInterfaze(base_url=api.base_url)
    with pytest.raises(openai.RateLimitError):
        list(model.stream(MESSAGES))
    with pytest.raises(openai.RateLimitError):
        [chunk async for chunk in model.astream(MESSAGES)]
    assert len(api.authorization) == DEFAULT_ATTEMPTS * 2


async def test_streamed_sync_daily_limit_is_found_by_body_with_a_short_wait(api: FakeApi) -> None:
    api.reply(429, DAILY_LIMIT_BODY, {"Retry-After": str(SHORT_WAIT)})
    with pytest.raises(openai.RateLimitError):
        list(ChatInterfaze(base_url=api.base_url).stream(MESSAGES))
    assert len(api.authorization) == 1


async def test_streamed_async_429_with_a_date_for_retry_after_keeps_the_default_retries(
    api: FakeApi, no_retry_sleep: None
) -> None:
    api.reply(429, DAILY_LIMIT_BODY, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"})
    with pytest.raises(openai.RateLimitError):
        [chunk async for chunk in ChatInterfaze(base_url=api.base_url).astream(MESSAGES)]
    assert len(api.authorization) == DEFAULT_ATTEMPTS
