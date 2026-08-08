from __future__ import annotations

import pytest
import respx
from interfaze import InterfazeError
from langchain_core.messages import HumanMessage

from langchain_interfaze import ChatInterfaze
from tests.unit_tests.conftest import BASIC, VIDEO_URL, last_body, mock_json


@respx.mock
def test_video_block_converted_to_file_part() -> None:
    route = mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "what happens in this clip?"},
            {"type": "video", "url": VIDEO_URL},
        ]
    )
    model.invoke([message])
    content = last_body(route)["messages"][-1]["content"]
    assert {"type": "file", "file": {"file_data": VIDEO_URL, "format": "video/mp4"}} in content


@respx.mock
def test_video_block_url_without_known_extension_omits_format() -> None:
    route = mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    model.invoke([HumanMessage(content=[{"type": "video", "url": "https://example.com/clip"}])])
    assert last_body(route)["messages"][-1]["content"][0]["file"] == {"file_data": "https://example.com/clip"}


@respx.mock
def test_video_block_base64_converted_to_file_part() -> None:
    route = mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    model.invoke([HumanMessage(content=[{"type": "video", "base64": "AAAA", "mime_type": "video/mp4"}])])
    content = last_body(route)["messages"][-1]["content"]
    assert content[0]["type"] == "file"
    assert content[0]["file"]["file_data"] == "data:video/mp4;base64,AAAA"


@respx.mock
def test_video_block_forwards_filename() -> None:
    route = mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    model.invoke(
        [HumanMessage(content=[{"type": "video", "url": VIDEO_URL, "extras": {"filename": "clip.mp4"}}])]
    )
    file = last_body(route)["messages"][-1]["content"][0]["file"]
    assert file["file_data"] == VIDEO_URL
    assert file["filename"] == "clip.mp4"


@respx.mock
def test_video_block_file_id_raises() -> None:
    model = ChatInterfaze(api_key="t")
    with pytest.raises(InterfazeError, match="file_id"):
        model.invoke([HumanMessage(content=[{"type": "video", "file_id": "file-123"}])])


@respx.mock
def test_video_block_missing_source_raises() -> None:
    model = ChatInterfaze(api_key="t")
    with pytest.raises(InterfazeError, match="requires one of"):
        model.invoke([HumanMessage(content=[{"type": "video"}])])


@respx.mock
def test_file_id_rejected_on_any_block() -> None:
    model = ChatInterfaze(api_key="k")
    with pytest.raises(InterfazeError, match="file_id"):
        model.invoke([HumanMessage(content=[{"type": "file", "file_id": "file-123"}])])


def test_scalar_header_values_are_stringified() -> None:
    model = ChatInterfaze(api_key="k", default_headers={"X-Retries": 3})  # ty:ignore[invalid-argument-type]
    assert model.default_headers == {"x-retries": "3"}


@respx.mock
def test_file_id_rejected_in_the_openai_native_nesting() -> None:
    model = ChatInterfaze(api_key="k")
    with pytest.raises(InterfazeError, match="file_id"):
        model.invoke([HumanMessage(content=[{"type": "file", "file": {"file_id": "file-abc"}}])])


@respx.mock
def test_empty_mime_type_falls_through_to_the_default() -> None:
    route = mock_json(BASIC)
    ChatInterfaze(api_key="t").invoke(
        [HumanMessage(content=[{"type": "video", "base64": "AAAA", "mime_type": ""}])]
    )
    part = last_body(route)["messages"][-1]["content"][0]
    assert part["file"] == {"file_data": "data:video/mp4;base64,AAAA", "format": "video/mp4"}
