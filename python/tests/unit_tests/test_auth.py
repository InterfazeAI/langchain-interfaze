from __future__ import annotations

from typing import Any

import pytest
import respx
from openai import AsyncOpenAI, Omit, OpenAI
from pydantic import SecretStr

from interfaze_langchain import ChatInterfaze
from tests.unit_tests.api_mock import (
    ALL_PATHS,
    MESSAGES,
    Call,
    bearer,
    mock_api,
    sent_authorization,
)

ENV_KEY = "env-key-1"
EXPLICIT_KEY = "explicit-key-2"
FIELD_KEY = "field-key-3"
OPENAI_KEY = "openai-key-4"
OPENAI_ADMIN_KEY = "openai-admin-key-7"
NEW_KEY = "new-key-5"
CALLER_TOKEN = "caller-token-6"
CALLER_AUTH = bearer(CALLER_TOKEN)

PATHS = pytest.mark.parametrize("path", ALL_PATHS.values(), ids=list(ALL_PATHS))


def _configure(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> None:
    [monkeypatch.setenv(name, value) for name, value in env.items()]


def _keyless(id_: str, kwargs: dict[str, Any], env: dict[str, str]) -> Any:
    return pytest.param(kwargs, env, id=id_)


def _keyed(id_: str, kwargs: dict[str, Any], env: dict[str, str], expected: str) -> Any:
    return pytest.param(kwargs, env, expected, id=id_)


KEYLESS = [
    _keyless("no key", {}, {}),
    _keyless("only OPENAI_API_KEY", {}, {"OPENAI_API_KEY": OPENAI_KEY}),
    _keyless("empty api_key", {"api_key": ""}, {}),
    _keyless("space api_key", {"api_key": " "}, {}),
    _keyless("space-newline api_key", {"api_key": " \n"}, {}),
    _keyless("blank SecretStr api_key", {"api_key": SecretStr(" ")}, {}),
    _keyless("blank openai_api_key", {"openai_api_key": " "}, {}),
    _keyless("empty SecretStr openai_api_key", {"openai_api_key": SecretStr("")}, {}),
    _keyless("blank env", {}, {"INTERFAZE_API_KEY": " \n"}),
    _keyless("empty env", {}, {"INTERFAZE_API_KEY": ""}),
    _keyless("blank explicit and blank env", {"api_key": " "}, {"INTERFAZE_API_KEY": " "}),
    _keyless("blank key and OPENAI_API_KEY", {"api_key": " "}, {"OPENAI_API_KEY": OPENAI_KEY}),
]

BOTH_ENV = {"INTERFAZE_API_KEY": ENV_KEY, "OPENAI_API_KEY": OPENAI_KEY}
KEYED = [
    _keyed("env key", {}, {"INTERFAZE_API_KEY": ENV_KEY}, ENV_KEY),
    _keyed("env key beside OPENAI_API_KEY", {}, BOTH_ENV, ENV_KEY),
    _keyed("env key is trimmed", {}, {"INTERFAZE_API_KEY": f"  {ENV_KEY}\n"}, ENV_KEY),
    _keyed("explicit beats env", {"api_key": EXPLICIT_KEY}, BOTH_ENV, EXPLICIT_KEY),
    _keyed("explicit is trimmed", {"api_key": f" {EXPLICIT_KEY}\n"}, {}, EXPLICIT_KEY),
    _keyed("api_key SecretStr", {"api_key": SecretStr(EXPLICIT_KEY)}, {}, EXPLICIT_KEY),
    _keyed("openai_api_key str", {"openai_api_key": FIELD_KEY}, {}, FIELD_KEY),
    _keyed("openai_api_key SecretStr", {"openai_api_key": SecretStr(FIELD_KEY)}, {}, FIELD_KEY),
    _keyed("openai_api_key beats env", {"openai_api_key": FIELD_KEY}, BOTH_ENV, FIELD_KEY),
    _keyed("empty api_key sends env", {"api_key": ""}, BOTH_ENV, ENV_KEY),
    _keyed("space api_key sends env", {"api_key": " "}, BOTH_ENV, ENV_KEY),
    _keyed("space-newline api_key sends env", {"api_key": " \n"}, BOTH_ENV, ENV_KEY),
    _keyed("blank openai_api_key sends env", {"openai_api_key": " "}, BOTH_ENV, ENV_KEY),
]


def _leaks_openai_key(route: respx.Route) -> bool:
    return any(OPENAI_KEY in value for call in route.calls for value in call.request.headers.values())


@respx.mock
@PATHS
@pytest.mark.parametrize(("kwargs", "env"), KEYLESS)
async def test_keyless_request_has_no_authorization_header(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], env: dict[str, str], path: Call
) -> None:
    route = mock_api()
    _configure(monkeypatch, env)
    await path(ChatInterfaze(**kwargs))
    assert route.called
    assert sent_authorization(route) == [[]] * route.call_count
    assert not _leaks_openai_key(route)


@respx.mock
@PATHS
@pytest.mark.parametrize(("kwargs", "env", "expected"), KEYED)
async def test_key_is_sent_as_the_only_authorization_header(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], env: dict[str, str], expected: str, path: Call
) -> None:
    route = mock_api()
    _configure(monkeypatch, env)
    await path(ChatInterfaze(**kwargs))
    assert route.called
    assert sent_authorization(route) == [[bearer(expected)]] * route.call_count
    assert not _leaks_openai_key(route)


@respx.mock
async def test_key_callable_passes_through_untouched() -> None:
    route = mock_api()
    model = ChatInterfaze(openai_api_key=lambda: EXPLICIT_KEY)
    model.invoke(MESSAGES)
    await model.ainvoke(MESSAGES)
    assert sent_authorization(route) == [[bearer(EXPLICIT_KEY)]] * 2


CALLER_CASES = ["Authorization", "authorization", "AUTHORIZATION"]
KEY_SETUPS = [
    _keyless("no key", {}, {}),
    _keyless("api_key", {"api_key": EXPLICIT_KEY}, {}),
    _keyless("env key", {}, {"INTERFAZE_API_KEY": ENV_KEY}),
]


@respx.mock
@PATHS
@pytest.mark.parametrize(("kwargs", "env"), KEY_SETUPS)
@pytest.mark.parametrize("case", CALLER_CASES)
async def test_caller_default_authorization_header_is_the_only_one_sent(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], env: dict[str, str], case: str, path: Call
) -> None:
    route = mock_api()
    _configure(monkeypatch, env)
    await path(ChatInterfaze(default_headers={case: CALLER_AUTH}, **kwargs))
    assert route.called
    assert sent_authorization(route) == [[CALLER_AUTH]] * route.call_count


@respx.mock
@pytest.mark.parametrize("kwargs", [{}, {"api_key": EXPLICIT_KEY}], ids=["no key", "api_key"])
@pytest.mark.parametrize("case", CALLER_CASES)
async def test_caller_request_authorization_header_is_the_only_one_sent(
    kwargs: dict[str, Any], case: str
) -> None:
    route = mock_api()
    model = ChatInterfaze(**kwargs)
    model.invoke(MESSAGES, extra_headers={case: CALLER_AUTH})
    await model.ainvoke(MESSAGES, extra_headers={case: CALLER_AUTH})
    model.root_client.models.list(extra_headers={case: CALLER_AUTH})
    await model.root_async_client.models.list(extra_headers={case: CALLER_AUTH})
    assert sent_authorization(route) == [[CALLER_AUTH]] * 4


@respx.mock
@pytest.mark.parametrize("case", CALLER_CASES)
async def test_request_authorization_header_beats_default_header(case: str) -> None:
    route = mock_api()
    model = ChatInterfaze(default_headers={"Authorization": bearer("default-token")})
    model.root_client.models.list(extra_headers={case: CALLER_AUTH})
    await model.root_async_client.models.list(extra_headers={case: CALLER_AUTH})
    assert sent_authorization(route) == [[CALLER_AUTH]] * 2


@respx.mock
async def test_omitted_request_header_leaves_the_caller_default_header_as_the_only_one() -> None:
    route = mock_api()
    model = ChatInterfaze(api_key=EXPLICIT_KEY, default_headers={"authorization": CALLER_AUTH})
    model.root_client.models.list(extra_headers={"AUTHORIZATION": Omit()})
    await model.root_async_client.models.list(extra_headers={"AUTHORIZATION": Omit()})
    assert sent_authorization(route) == [[CALLER_AUTH]] * 2


@respx.mock
@pytest.mark.parametrize("derive", ["with_options", "copy"])
async def test_new_key_on_a_keyless_client_is_sent(derive: str) -> None:
    route = mock_api()
    model = ChatInterfaze()
    getattr(model.root_client, derive)(api_key=NEW_KEY).models.list()
    await getattr(model.root_async_client, derive)(api_key=NEW_KEY).models.list()
    assert sent_authorization(route) == [[bearer(NEW_KEY)]] * 2


@respx.mock
async def test_client_derived_without_a_new_key_stays_keyless() -> None:
    route = mock_api()
    ChatInterfaze().root_client.with_options(max_retries=0).copy(timeout=5).models.list()
    assert sent_authorization(route) == [[]]


def test_raw_clients_exist_without_a_key() -> None:
    model = ChatInterfaze()
    assert None not in (model.root_client, model.client, model.root_async_client, model.async_client)


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
async def test_keyless_model_serializes_and_still_sends_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    route = mock_api()
    monkeypatch.setenv("OPENAI_API_KEY", OPENAI_KEY)
    model = ChatInterfaze()
    assert model.model_dump_json()
    assert model.openai_api_key == SecretStr("")
    model.invoke(MESSAGES)
    model.root_client.with_options(timeout=5).models.list()
    assert sent_authorization(route) == [[], []]
    assert not _leaks_openai_key(route)


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
    model.root_client.with_options(timeout=5).admin.organization.admin_api_keys.list()
    await model.root_async_client.admin.organization.admin_api_keys.list()
    await model.root_async_client.copy(timeout=5).admin.organization.admin_api_keys.list()
    assert route.call_count == 4
    assert sent_authorization(route) == [[]] * 4
    assert not any(
        OPENAI_ADMIN_KEY in value for call in route.calls for value in call.request.headers.values()
    )
