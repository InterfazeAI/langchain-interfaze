from __future__ import annotations

from typing import Any

import httpx
import openai
import pytest
import respx
from langchain_core.messages import HumanMessage

from interfaze_langchain import ChatInterfaze
from tests.unit_tests.conftest import BASIC

API_HOST = "api.interfaze.ai"
EXPLICIT_KEY = "explicit-key-2"
OPENAI_KEY = "openai-key-4"
MESSAGES = [HumanMessage("hi")]
SIGNUP_HINT = (
    "Get a free API key at https://interfaze.ai and send it as `Authorization: Bearer YOUR_API_KEY`."
)
INVALID_KEY_BODY = {
    "error": {
        "message": "The API key provided is not valid",
        "type": "permission_error",
        "code": "insufficient_quota",
        "request_id": "req-test",
    }
}
DAILY_LIMIT_BODY = {
    "error": {
        "message": f"Keyless daily limit of 1000 requests reached. {SIGNUP_HINT}",
        "type": "rate_limit_error",
        "code": "keyless_daily_limit_exceeded",
        "request_id": "req-test",
    }
}
DAILY_LIMIT_HEADERS = {"Retry-After": "30", "x-should-retry": "false"}
INFLIGHT_BODY = {
    "error": {
        "message": f"Too many keyless requests in progress (limit 5). {SIGNUP_HINT}",
        "type": "rate_limit_error",
        "code": "keyless_inflight_limit_exceeded",
        "request_id": "req-test",
    }
}
INFLIGHT_HEADERS = {"Retry-After": "5"}


def mock_status(status: int, body: dict[str, Any], headers: dict[str, str] | None = None) -> respx.Route:
    return respx.route(host=API_HOST).mock(return_value=httpx.Response(status, json=body, headers=headers))


@respx.mock
@pytest.mark.parametrize(
    ("kwargs", "sent"),
    [({}, []), ({"api_key": EXPLICIT_KEY}, [f"Bearer {EXPLICIT_KEY}"])],
    ids=["no key", "api_key"],
)
async def test_401_reaches_the_caller_after_one_request(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], sent: list[str], no_retry_sleep: None
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    route = mock_status(401, INVALID_KEY_BODY)
    with pytest.raises(openai.AuthenticationError) as caught:
        ChatInterfaze(**kwargs).invoke(MESSAGES)
    assert [call.request.headers.get_list("authorization") for call in route.calls] == [sent]
    assert caught.value.status_code == 401
    assert caught.value.code == "insufficient_quota"
    assert caught.value.type == "permission_error"
    assert caught.value.response.json() == INVALID_KEY_BODY


@respx.mock
async def test_429_that_says_not_to_retry_reaches_the_caller_after_one_request(no_retry_sleep: None) -> None:
    route = mock_status(429, DAILY_LIMIT_BODY, DAILY_LIMIT_HEADERS)
    with pytest.raises(openai.RateLimitError) as caught:
        await ChatInterfaze().ainvoke(MESSAGES)
    assert route.call_count == 1
    assert caught.value.code == "keyless_daily_limit_exceeded"
    assert SIGNUP_HINT in caught.value.message


@respx.mock
async def test_inflight_429_is_retried(no_retry_sleep: None) -> None:
    route = respx.route(host=API_HOST).mock(
        side_effect=[
            httpx.Response(429, json=INFLIGHT_BODY, headers=INFLIGHT_HEADERS),
            httpx.Response(200, json=BASIC),
        ]
    )
    assert ChatInterfaze().invoke(MESSAGES).content == "Hi!"
    assert route.call_count == 2
