from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from tests.integration_tests.conftest import FAST, SLOW, chat, fresh_chat, requires_key, text_of

pytestmark = requires_key


@pytest.fixture(scope="module")
def llm():
    return chat()


# --- text -----------------------------------------------------------------


@pytest.mark.timeout(FAST)
def test_text_gen_story(llm) -> None:
    assert len(text_of(llm.invoke("Write a short story about a robot learning to paint"))) > 50


@pytest.mark.timeout(FAST)
def test_text_gen_story_with_system(llm) -> None:
    res = llm.invoke(
        [
            SystemMessage("You are a helpful assistant."),
            HumanMessage("Write a short story about a robot learning to paint"),
        ]
    )
    assert len(text_of(res)) > 50


@pytest.mark.timeout(FAST)
def test_text_capital(llm) -> None:
    assert "paris" in text_of(llm.invoke("What is the capital of France? Answer in one word.")).lower()


@pytest.mark.timeout(FAST)
def test_interfaze_envelope(llm) -> None:
    res = llm.invoke("Hello")
    assert res.response_metadata["model_provider"] == "interfaze"
    assert isinstance(res.response_metadata["vcache"], bool)
    assert res.usage_metadata["total_tokens"] > 0


# --- structured output ----------------------------------------------------


class Weather(BaseModel):
    city: str
    temperature_celsius: float
    condition: str


class Founder(BaseModel):
    name: str


class CapitalPop(BaseModel):
    city: str
    population_millions: float


class Capital(BaseModel):
    capital: str


@pytest.mark.timeout(FAST)
def test_structured_weather(llm) -> None:
    out = llm.with_structured_output(Weather).invoke("What is the current weather in Tokyo?")
    assert out.city
    assert isinstance(out.temperature_celsius, float)
    assert out.condition


@pytest.mark.timeout(FAST)
def test_structured_founder(llm) -> None:
    assert llm.with_structured_output(Founder).invoke("Who is the founder of JigsawStack?").name


@pytest.mark.timeout(FAST)
def test_structured_capital_pop(llm) -> None:
    out = llm.with_structured_output(CapitalPop).invoke(
        "What is the capital of France and its approximate metro population in millions?"
    )
    assert "paris" in out.city.lower()
    assert isinstance(out.population_millions, float)


@pytest.mark.timeout(FAST)
def test_structured_json_no_fences(llm) -> None:
    out = llm.with_structured_output(Capital, include_raw=True).invoke(
        "Return ONLY a JSON object (no markdown fences) with key 'capital' set to the capital of France."
    )
    assert "```" not in text_of(out["raw"])
    assert "paris" in out["parsed"].capital.lower()


# --- reasoning ------------------------------------------------------------


@pytest.mark.timeout(FAST)
def test_reasoning_math() -> None:
    # The semantic cache replays a stored answer without its <think> block.
    res = fresh_chat().invoke("What is 25 * 47?", reasoning_effort="high")
    assert "1175" in text_of(res)
    assert res.response_metadata.get("reasoning")
    assert "<think>" not in text_of(res)


# --- runnable surface -----------------------------------------------------


@pytest.mark.timeout(FAST)
def test_lcel_chain(llm) -> None:
    chain = ChatPromptTemplate.from_template("Translate to {lang}: {text}") | llm | StrOutputParser()
    assert chain.invoke({"lang": "French", "text": "Hello"})


@pytest.mark.timeout(SLOW)
def test_batch(llm) -> None:
    out = llm.batch(["Summarize the colour blue in one sentence.", "Name one planet.", "What is 2+2?"])
    assert len(out) == 3
    assert all(text_of(m) for m in out)


@pytest.mark.timeout(FAST)
async def test_ainvoke(llm) -> None:
    res = await llm.ainvoke("What is the capital of France? Answer in one word.")
    assert "paris" in text_of(res).lower()
    assert res.response_metadata["model_provider"] == "interfaze"
