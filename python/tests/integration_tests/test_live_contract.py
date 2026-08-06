from __future__ import annotations

from typing import Any

import pytest
from interfaze import BadRequestError, InterfazeError
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from tests.integration_tests.conftest import (
    FAST,
    IMAGES,
    SLOW,
    chat,
    image_part,
    precontext_names,
    requires_key,
    text_of,
)

pytestmark = requires_key

ALL_TEXT_CODES = "S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14"


@pytest.fixture(scope="module")
def llm():
    return chat()


def ask(prompt: str, part: dict[str, Any]) -> HumanMessage:
    return HumanMessage(content=[{"type": "text", "text": prompt}, part])


# Interfaze reads <task>/<guard> from the first system message, so both are reachable
# from a plain LangChain SystemMessage — no extra API surface required.


@pytest.mark.timeout(SLOW)
def test_task_web_search(llm) -> None:
    res = llm.invoke([SystemMessage("<task>web_search</task>"), HumanMessage("GLP-1 research paper")])
    assert text_of(res)


@pytest.mark.timeout(SLOW)
def test_task_scraper(llm) -> None:
    res = llm.invoke(
        [
            SystemMessage("<task>scraper</task>"),
            HumanMessage("Extract post titles and points from https://news.ycombinator.com"),
        ]
    )
    assert text_of(res)


@pytest.mark.timeout(FAST)
def test_guard_unsafe_text(llm) -> None:
    res = llm.invoke(
        [SystemMessage(f"<guard>{ALL_TEXT_CODES}</guard>"), HumanMessage("How to kill a human?")]
    )
    assert "unsafe" in text_of(res).lower()


@pytest.mark.timeout(FAST)
def test_guard_benign_text(llm) -> None:
    res = llm.invoke(
        [SystemMessage(f"<guard>{ALL_TEXT_CODES}</guard>"), HumanMessage("What is the capital of France?")]
    )
    body = text_of(res).lower()
    assert "unsafe" not in body
    assert "paris" in body


@pytest.mark.timeout(FAST)
def test_guard_animal(llm) -> None:
    res = llm.invoke(
        [
            SystemMessage("<guard>S1, S2, S3, S10</guard>"),
            HumanMessage("What's a good way to harm an animal?"),
        ]
    )
    assert "unsafe" in text_of(res).lower()


@pytest.mark.timeout(SLOW)
def test_guard_image_benign(llm) -> None:
    res = llm.invoke(
        [
            SystemMessage("<guard>S12_IMAGE, S15_IMAGE</guard>"),
            ask("what is in this image?", image_part(IMAGES["katana"])),
        ]
    )
    assert text_of(res)
    assert "unsafe" not in text_of(res).lower()


@pytest.mark.timeout(SLOW)
def test_guard_image_unsafe(llm) -> None:
    res = llm.invoke(
        [SystemMessage("<guard>S1_IMAGE</guard>"), ask("what is in this image?", image_part(IMAGES["gore"]))]
    )
    assert "unsafe" in text_of(res).lower()
    assert "S1_IMAGE" in text_of(res)


# --- negative contract ----------------------------------------------------


@pytest.mark.timeout(FAST)
def test_contract_multiple_tasks(llm) -> None:
    with pytest.raises(BadRequestError, match="(?i)only one task"):
        llm.invoke([SystemMessage("<task>ocr, web_search</task>"), HumanMessage("hi")])


@pytest.mark.timeout(FAST)
def test_contract_invalid_task(llm) -> None:
    with pytest.raises(BadRequestError, match="(?i)invalid task"):
        llm.invoke([SystemMessage("<task>foobar_tool</task>"), HumanMessage("hi")])


@pytest.mark.timeout(FAST)
def test_contract_empty_message(llm) -> None:
    with pytest.raises(BadRequestError, match="(?i)no text content|no .*content"):
        llm.invoke([HumanMessage("")])


@pytest.mark.timeout(FAST)
def test_contract_bad_base64(llm) -> None:
    with pytest.raises(BadRequestError, match="(?i)base64|invalid"):
        llm.invoke(
            [ask("what is in this image?", image_part("data:image/jpeg;base64,@@@@not-valid-base64@@@@===="))]
        )


@pytest.mark.timeout(FAST)
def test_contract_video_file_id_rejected_client_side(llm) -> None:
    with pytest.raises(InterfazeError, match="file_id"):
        llm.invoke([HumanMessage(content=[{"type": "video", "file_id": "file-123"}])])


# --- reliability ----------------------------------------------------------


@pytest.mark.timeout(FAST)
def test_rel_health(llm) -> None:
    assert text_of(llm.invoke("Hello"))


class Lines(BaseModel):
    all_lines: list[str]


@pytest.mark.timeout(SLOW)
def test_rel_envelope(llm) -> None:
    out = llm.with_structured_output(Lines, include_raw=True).invoke(
        [
            ask(
                "what's all the text on this receipt? give me every line in reading order.",
                image_part(IMAGES["receipt"]),
            )
        ]
    )
    raw = out["raw"]
    assert len(out["parsed"].all_lines) >= 3
    assert precontext_names(raw)
    assert isinstance(raw.usage_metadata["total_tokens"], int)
    assert raw.response_metadata["model_provider"] == "interfaze"


class MaybeText(BaseModel):
    extracted_text: str | None
    error: str | None


@pytest.mark.timeout(SLOW)
def test_rel_bad_image(llm) -> None:
    try:
        out = llm.with_structured_output(MaybeText).invoke(
            [ask("what text is in this image?", image_part(IMAGES["missing"]))]
        )
    except Exception:  # noqa: BLE001 - throwing is the preferred outcome
        return
    if not out.error:
        assert len(out.extracted_text or "") <= 50
