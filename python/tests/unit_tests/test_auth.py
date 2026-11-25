from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import patch

import httpx
import pytest
import respx
from langchain_core.messages import HumanMessage
from openai import AsyncOpenAI, OpenAI
from pydantic import SecretStr

from interfaze_langchain import ChatInterfaze
from tests.unit_tests.conftest import BASIC, _sse_bytes, chunk

API_HOST = "api.interfaze.ai"
ENV_KEY = "env-key-1"
EXPLICIT_KEY = "explicit-key-2"
FIELD_KEY = "field-key-3"
OPENAI_KEY = "openai-key-4"
NEW_KEY = "new-key-5"
OPENAI_ADMIN_KEY = "openai-admin-key-7"
CLIENT_TIMEOUT = 5
BOTH_ENV = {"INTERFAZE_API_KEY": ENV_KEY, "OPENAI_API_KEY": OPENAI_KEY}
MESSAGES = [HumanMessage("hi")]

_MODELS = {"object": "list", "data": []}
_STREAM = [chunk({"content": "Hi"}), chunk({}, finish_reason="stop")]

Call = Callable[[ChatInterfaze], Awaitable[object]]


def bearer(token: str) -> str:
    return f"Bearer {token}"


def _answer(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/chat/completions") and json.loads(request.content).get("stream"):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_bytes(_STREAM))
    body = BASIC if path.endswith("/chat/completions") else _MODELS
    return httpx.Response(200, json=body)


def mock_api() -> respx.Route:
    """Answer every request to the API host, whatever the path."""
    return respx.route(host=API_HOST).mock(side_effect=_answer)


def sent_authorization(route: respx.Route) -> list[list[str]]:
    """The Authorization values of each request, one list per request."""
    return [call.request.headers.get_list("authorization") for call in route.calls]


def _leaks_openai_key(route: respx.Route) -> bool:
    return any(OPENAI_KEY in value for call in route.calls for value in call.request.headers.values())


async def _invoke(model: ChatInterfaze) -> object:
    return model.invoke(MESSAGES)


async def _ainvoke(model: ChatInterfaze) -> object:
    return await model.ainvoke(MESSAGES)


async def _stream(model: ChatInterfaze) -> object:
    return list(model.stream(MESSAGES))


async def _astream(model: ChatInterfaze) -> object:
    return [part async for part in model.astream(MESSAGES)]


async def _models_list(model: ChatInterfaze) -> object:
    return model.root_client.models.list()


async def _async_models_list(model: ChatInterfaze) -> object:
    return await model.root_async_client.models.list()


async def _with_options(model: ChatInterfaze) -> object:
    return model.root_client.with_options(timeout=CLIENT_TIMEOUT).models.list()


async def _async_copy(model: ChatInterfaze) -> object:
    return await model.root_async_client.copy(timeout=CLIENT_TIMEOUT).models.list()


CALL_PATHS: dict[str, Call] = {
    "invoke": _invoke,
    "ainvoke": _ainvoke,
    "stream": _stream,
    "astream": _astream,
    "models.list": _models_list,
    "async models.list": _async_models_list,
    "with_options": _with_options,
    "async copy": _async_copy,
}


@respx.mock
@pytest.mark.parametrize("path", CALL_PATHS.values(), ids=CALL_PATHS.keys())
async def test_keyless_request_has_no_authorization_header_on_every_call_path(path: Call) -> None:
    route = mock_api()
    with patch.dict(os.environ, {"OPENAI_API_KEY": OPENAI_KEY}):
        await path(ChatInterfaze())
    assert route.called
    assert sent_authorization(route) == [[]] * route.call_count
    assert not _leaks_openai_key(route)


@respx.mock
@pytest.mark.parametrize(
    ("kwargs", "env"),
    [
        pytest.param({"api_key": " \n"}, {"INTERFAZE_API_KEY": " "}, id="blank api_key and blank env"),
        pytest.param({"openai_api_key": SecretStr(" ")}, {}, id="blank SecretStr openai_api_key"),
    ],
)
async def test_blank_key_counts_as_no_key(kwargs: dict[str, Any], env: dict[str, str]) -> None:
    route = mock_api()
    with patch.dict(os.environ, env):
        ChatInterfaze(**kwargs).invoke(MESSAGES)
    assert sent_authorization(route) == [[]]


@respx.mock
@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({}, ENV_KEY, id="env key"),
        pytest.param({"api_key": f" {EXPLICIT_KEY}\n"}, EXPLICIT_KEY, id="trimmed api_key beats env"),
        pytest.param({"openai_api_key": SecretStr(FIELD_KEY)}, FIELD_KEY, id="openai_api_key beats env"),
        pytest.param({"api_key": " \n"}, ENV_KEY, id="blank api_key sends env"),
    ],
)
async def test_key_is_sent_as_the_only_authorization_header(kwargs: dict[str, Any], expected: str) -> None:
    route = mock_api()
    with patch.dict(os.environ, BOTH_ENV):
        ChatInterfaze(**kwargs).invoke(MESSAGES)
    assert sent_authorization(route) == [[bearer(expected)]]
    assert not _leaks_openai_key(route)


@respx.mock
async def test_key_callable_passes_through_untouched() -> None:
    route = mock_api()
    model = ChatInterfaze(openai_api_key=lambda: EXPLICIT_KEY)
    model.invoke(MESSAGES)
    await model.ainvoke(MESSAGES)
    assert sent_authorization(route) == [[bearer(EXPLICIT_KEY)]] * 2


@respx.mock
@pytest.mark.parametrize("derive", ["with_options", "copy"])
async def test_new_key_on_a_keyless_client_is_sent(derive: str) -> None:
    route = mock_api()
    model = ChatInterfaze()
    getattr(model.root_client, derive)(api_key=NEW_KEY).models.list()
    await getattr(model.root_async_client, derive)(api_key=NEW_KEY).models.list()
    assert sent_authorization(route) == [[bearer(NEW_KEY)]] * 2


def test_callers_own_clients_keep_their_class() -> None:
    sync_client = OpenAI(api_key=EXPLICIT_KEY)
    async_client = AsyncOpenAI(api_key=EXPLICIT_KEY)
    model = ChatInterfaze(
        client=sync_client.chat.completions,
        root_client=sync_client,
        async_client=async_client.chat.completions,
        root_async_client=async_client,
    )
    assert model.root_client is sync_client
    assert model.root_async_client is async_client
    assert type(sync_client) is OpenAI
    assert type(async_client) is AsyncOpenAI


@respx.mock
async def test_keyless_model_serializes_and_sends_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    route = mock_api()
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    model = ChatInterfaze()
    assert model.model_dump_json()
    assert model.openai_api_key == SecretStr("")
    model.invoke(MESSAGES)
    assert sent_authorization(route) == [[]]


@respx.mock
@pytest.mark.skipif(not hasattr(OpenAI, "admin"), reason="This openai has no admin resource.")
@pytest.mark.parametrize("kwargs", [{}, {"api_key": EXPLICIT_KEY}], ids=["no key", "api_key"])
async def test_openai_admin_key_is_never_sent(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any]
) -> None:
    route = mock_api()
    monkeypatch.setenv("OPENAI_ADMIN_KEY", OPENAI_ADMIN_KEY)
    model = ChatInterfaze(**kwargs)
    model.root_client.admin.organization.admin_api_keys.list()
    await model.root_async_client.admin.organization.admin_api_keys.list()
    assert sent_authorization(route) == [[]] * 2
    assert not any(
        OPENAI_ADMIN_KEY in value for call in route.calls for value in call.request.headers.values()
    )


@respx.mock
async def test_user_agent_names_the_interfaze_clients() -> None:
    route = mock_api()
    model = ChatInterfaze()
    model.root_client.models.list()
    await model.root_async_client.models.list()
    sync_agent, async_agent = (call.request.headers["user-agent"] for call in route.calls)
    assert sync_agent.startswith("InterfazeOpenAI/Python")
    assert async_agent.startswith("AsyncInterfazeOpenAI/Python")
