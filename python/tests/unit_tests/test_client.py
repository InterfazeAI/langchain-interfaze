from __future__ import annotations

from typing import Any

import pytest
import respx
from interfaze import INTERFAZE_BASE_URL, INTERFAZE_MODEL, InterfazeError
from langchain_core.messages import HumanMessage

from langchain_interfaze import ChatInterfaze
from tests.unit_tests.conftest import BASIC, chunk, last_body, mock_json, mock_sse


# defaults
def test_defaults_point_at_interfaze() -> None:
    model = ChatInterfaze(api_key="t")
    assert model.openai_api_base == INTERFAZE_BASE_URL
    assert model.model_name == INTERFAZE_MODEL


def test_defaults_overridable() -> None:
    model = ChatInterfaze(api_key="t", base_url="https://example.com/v1", model="other-model")
    assert model.openai_api_base == "https://example.com/v1"
    assert model.model_name == "other-model"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERFAZE_API_KEY", raising=False)
    with pytest.raises(InterfazeError, match="Missing API key"):
        ChatInterfaze()


def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERFAZE_API_KEY", "env-key")
    model = ChatInterfaze()
    assert model.openai_api_key is not None


def test_defaults_to_long_timeout_but_respects_override() -> None:
    assert ChatInterfaze(api_key="t").request_timeout == 900.0
    assert ChatInterfaze(api_key="t", timeout=30).request_timeout == 30


def test_never_routes_to_the_responses_api() -> None:
    # `reasoning=` would otherwise flip ChatOpenAI over to /v1/responses.
    assert ChatInterfaze(api_key="t", reasoning={"summary": "auto"}).use_responses_api is False


@respx.mock
def test_reasoning_kwarg_is_folded_into_reasoning_effort() -> None:
    route = mock_json(BASIC)
    model = ChatInterfaze(api_key="t", reasoning={"effort": "high", "summary": "auto"})
    assert model.invoke([HumanMessage("hi")]).content == "Hi!"
    body = last_body(route)
    assert "reasoning" not in body
    assert body["reasoning_effort"] == "high"


@respx.mock
def test_reasoning_kwarg_without_effort_is_dropped() -> None:
    route = mock_json(BASIC)
    ChatInterfaze(api_key="t", reasoning={"summary": "auto"}).invoke([HumanMessage("hi")])
    body = last_body(route)
    assert "reasoning" not in body
    assert "reasoning_effort" not in body


# control-plane headers
def test_control_headers() -> None:
    model = ChatInterfaze(
        api_key="t",
        show_additional_info=True,
        bypass_moa=True,
        bypass_cache=True,
        default_headers={"x-custom": "1"},
    )
    assert model.default_headers == {
        "x-custom": "1",
        "x-show-additional-info": "true",
        "x-interfaze-bypass-moa": "true",
        "x-interfaze-bypass-cache": "true",
    }


def test_no_control_headers_by_default() -> None:
    assert ChatInterfaze(api_key="t").default_headers is None


@respx.mock
def test_streaming_asks_for_usage() -> None:
    # langchain-openai only auto-enables this for OpenAI's own base URL.
    route = mock_sse([chunk({"content": "hi"}), chunk({}, finish_reason="stop")])
    list(ChatInterfaze(api_key="t").stream([HumanMessage("hi")]))
    assert last_body(route)["stream_options"] == {"include_usage": True}


def test_cache_key_distinguishes_header_values() -> None:
    a = ChatInterfaze(api_key="k", default_headers={"x-tenant": "a"})
    b = ChatInterfaze(api_key="k", default_headers={"x-tenant": "b"})
    assert a._get_llm_string() != b._get_llm_string()


def test_control_header_replaces_a_differently_cased_one() -> None:
    model = ChatInterfaze(
        api_key="k", bypass_cache=True, default_headers={"X-Interfaze-Bypass-Cache": "false"}
    )
    assert model.default_headers == {"x-interfaze-bypass-cache": "true"}


@pytest.mark.parametrize(
    ("model_kwargs", "call_kwargs", "expected"),
    [
        ({}, {"reasoning": {"effort": "high"}, "reasoning_effort": "low"}, "high"),
        ({"reasoning": {"effort": "low"}}, {"reasoning_effort": "high"}, "high"),
        ({"reasoning_effort": "low"}, {"reasoning": {"effort": "high"}}, "high"),
        ({"reasoning_effort": "on"}, {}, "on"),
    ],
)
def test_reasoning_effort_precedence(
    model_kwargs: dict[str, Any], call_kwargs: dict[str, Any], expected: str
) -> None:
    """Same ladder as the JS package: a per-call value always beats a model-level one."""
    model = ChatInterfaze(api_key="k", **model_kwargs)
    assert model._get_request_payload([HumanMessage("x")], **call_kwargs)["reasoning_effort"] == expected


def test_cache_key_does_not_publish_header_values() -> None:
    model = ChatInterfaze(api_key="k", default_headers={"x-tenant": "secret-tenant"})
    assert "secret-tenant" not in str(model._identifying_params)
    assert "secret-tenant" not in model._get_llm_string()


def test_cache_key_shows_the_flags_we_own() -> None:
    model = ChatInterfaze(api_key="k", bypass_cache=True)
    assert model._identifying_params["interfaze_headers"] == ["x-interfaze-bypass-cache=true"]


def test_cache_key_separates_two_api_keys() -> None:
    a = ChatInterfaze(api_key="sk_tenant_a")
    b = ChatInterfaze(api_key="sk_tenant_b")
    assert a._get_llm_string() != b._get_llm_string()
    assert "sk_tenant_a" not in a._get_llm_string()
