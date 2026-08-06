from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage

from tests.integration_tests.conftest import (
    FAST,
    IMAGES,
    SLOW,
    chat,
    fresh_chat,
    image_part,
    requires_key,
)

pytestmark = requires_key


class Tap(BaseCallbackHandler):
    def __init__(self) -> None:
        self.tokens: list[str] = []

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.tokens.append(token)


@pytest.fixture(scope="module")
def llm():
    return chat()


@pytest.mark.timeout(FAST)
def test_stream_haiku(llm) -> None:
    out = "".join(c.content for c in llm.stream("Write a haiku about coding") if isinstance(c.content, str))
    assert len(out) > 10
    assert "<think>" not in out
    assert "<precontext>" not in out


@pytest.mark.timeout(FAST)
def test_stream_capital(llm) -> None:
    out = "".join(
        c.content
        for c in llm.stream("What is the capital of France? Answer in one word.")
        if isinstance(c.content, str)
    )
    assert "paris" in out.lower()


@pytest.mark.timeout(FAST)
def test_stream_reasoning_hides_think_tags() -> None:
    chunks = list(fresh_chat().stream("Write a haiku about streaming data", reasoning_effort="high"))
    visible = "".join(c.content for c in chunks if isinstance(c.content, str))
    assert "<think>" not in visible
    assert "</think>" not in visible
    assert [c for c in chunks if c.additional_kwargs.get("reasoning")]


@pytest.mark.timeout(FAST)
def test_token_callbacks_never_see_side_channels() -> None:
    tap = Tap()
    list(
        fresh_chat().stream(
            "Write a haiku about streaming data",
            reasoning_effort="high",
            config={"callbacks": [tap]},
        )
    )
    assert "<think>" not in "".join(tap.tokens)


@pytest.mark.timeout(FAST)
async def test_astream_events_are_filtered() -> None:
    llm = fresh_chat()
    body = ""
    async for ev in llm.astream_events(
        "Write a haiku about streaming data", version="v2", reasoning_effort="high"
    ):
        if ev["event"] == "on_chat_model_stream":
            content = ev["data"]["chunk"].content
            if isinstance(content, str):
                body += content
    assert body
    assert "<think>" not in body


@pytest.mark.timeout(FAST)
async def test_astream_matches_sync() -> None:
    llm = chat()
    out = "".join(
        [c.content async for c in llm.astream("Write a haiku about coding") if isinstance(c.content, str)]
    )
    assert len(out) > 10
    assert "<think>" not in out


@pytest.mark.timeout(SLOW)
def test_streams_inline_precontext_when_enabled() -> None:
    verbose = chat(show_additional_info=True, bypass_cache=True)
    chunks = list(
        verbose.stream(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Where is this store located?"},
                        image_part(IMAGES["receipt"]),
                    ]
                )
            ]
        )
    )
    visible = "".join(c.content for c in chunks if isinstance(c.content, str))
    assert "<precontext>" not in visible
    precontext = [e for c in chunks for e in (c.additional_kwargs.get("precontext") or [])]
    assert precontext
