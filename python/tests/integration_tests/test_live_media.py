from __future__ import annotations

import re
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from tests.integration_tests.conftest import (
    FILES,
    SLOW,
    chat,
    file_part,
    precontext_names,
    requires_key,
    text_of,
    video_part,
)

pytestmark = [requires_key, pytest.mark.timeout(SLOW)]


@pytest.fixture(scope="module")
def llm():
    return chat()


def ask(prompt: str, part: dict[str, Any]) -> HumanMessage:
    return HumanMessage(content=[{"type": "text", "text": prompt}, part])


class Transcript(BaseModel):
    text: str


def test_stt_basic(llm) -> None:
    out = llm.with_structured_output(Transcript, include_raw=True).invoke(
        [ask("Transcribe the audio file", file_part(FILES["stt_short"], "stt_medical_short.mp4"))]
    )
    assert "amoxicillin" in out["parsed"].text.lower()
    assert re.search(r"stt|speech_to_text", " ".join(precontext_names(out["raw"])))


class Chunk(BaseModel):
    speaker_id: str
    text: str
    start_time: float
    end_time: float


class Diarized(BaseModel):
    full_text: str
    chunks: list[Chunk]
    number_of_speakers: int


def test_stt_diarization(llm) -> None:
    out = llm.with_structured_output(Diarized, include_raw=True).invoke(
        [
            ask(
                "Transcribe and identify the speakers in the audio file",
                file_part(FILES["stt_multi"], "stt_multispeaker.mp3"),
            )
        ]
    )
    parsed: Diarized = out["parsed"]
    assert parsed.number_of_speakers >= 2
    assert len(parsed.chunks) > 5
    assert precontext_names(out["raw"])


class Translated(BaseModel):
    translated_text: str
    original_language_code: str
    translated_language_code: str


def test_stt_translate(llm) -> None:
    out = llm.with_structured_output(Translated).invoke(
        f"Transcribe the audio file and translate it to chinese {FILES['stt_short']}"
    )
    assert out.translated_language_code.lower() in {"zh", "zh-cn", "zh-tw"}


class CallSummary(BaseModel):
    text: str
    summary: str
    intent: str


def test_stt_summary(llm) -> None:
    out = llm.with_structured_output(CallSummary).invoke(
        f"Transcribe the audio file and summarize it {FILES['stt_call']}"
    )
    assert out.text and out.summary and out.intent


def test_video_describe(llm) -> None:
    res = llm.invoke(
        [ask("Describe what happens in this video in one or two sentences.", video_part(FILES["video"]))]
    )
    body = text_of(res).lower()
    assert body
    assert re.search(r"rabbit|bunny|forest|tree|grass|animal|burrow|meadow|field|nature", body)
