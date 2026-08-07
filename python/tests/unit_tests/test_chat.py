from __future__ import annotations

import asyncio
from typing import Any

import respx
from langchain_core.messages import HumanMessage

from langchain_interfaze import ChatInterfaze
from tests.unit_tests.conftest import BASIC, CUSTOM_FIELDS, INLINE_TAGS, mock_json


@respx.mock
def test_custom_response_fields_surfaced() -> None:
    mock_json(CUSTOM_FIELDS)
    model = ChatInterfaze(api_key="t")
    result = model.invoke([HumanMessage("hi")])
    assert result.response_metadata["precontext"] == [{"name": "ocr", "result": {"extracted_text": "x"}}]
    assert result.response_metadata["reasoning"] == "because reasons"
    assert result.response_metadata["vcache"] is True
    assert result.additional_kwargs["precontext"] == [{"name": "ocr", "result": {"extracted_text": "x"}}]
    assert result.additional_kwargs["reasoning"] == "because reasons"
    assert result.additional_kwargs["vcache"] is True


@respx.mock
def test_response_without_precontext_or_reasoning_unaffected() -> None:
    mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    result = model.invoke([HumanMessage("hi")])
    assert "precontext" not in result.response_metadata
    assert "reasoning" not in result.response_metadata
    assert result.response_metadata["vcache"] is False
    assert result.content == "Hi!"


@respx.mock
def test_non_streaming_strips_inline_tags() -> None:
    mock_json(INLINE_TAGS)
    model = ChatInterfaze(api_key="t")
    result = model.invoke([HumanMessage("why is the sky blue?")])
    assert result.content == "The sky is blue."
    assert result.response_metadata["reasoning"] == "Rayleigh scattering."
    assert result.response_metadata["precontext"] == [{"name": "ocr", "result": {"x": 1}}]


@respx.mock
def test_async_invoke_surfaces_side_fields() -> None:
    mock_json(CUSTOM_FIELDS)
    model = ChatInterfaze(api_key="t")

    async def go() -> Any:
        return await model.ainvoke([HumanMessage("hi")])

    result = asyncio.run(go())
    assert result.response_metadata["precontext"][0]["name"] == "ocr"
    assert result.response_metadata["vcache"] is True
