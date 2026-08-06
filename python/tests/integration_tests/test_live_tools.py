from __future__ import annotations

import json
import re

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel

from tests.integration_tests.conftest import SLOW, chat, precontext_names, requires_key, text_of

pytestmark = [requires_key, pytest.mark.timeout(SLOW)]


@pytest.fixture(scope="module")
def llm():
    return chat()


LONG_TEXT = (
    "Interfaze is a new kind of AI platform built specifically for deterministic, developer-grade "
    "tasks. Unlike general-purpose large language models that excel at open-ended conversation but "
    "struggle with consistency, Interfaze focuses on the operations that real software systems "
    "depend on: extracting fields from documents, scraping structured data from arbitrary websites, "
    "transcribing audio with speaker labels, translating content across hundreds of languages while "
    "preserving meaning, detecting objects in images and GUI screenshots, forecasting time series "
    "without per-customer model training, and executing code in a sandboxed environment. Every "
    "capability is exposed through an OpenAI-compatible chat completions API so existing tooling "
    "works without modification."
)

SERIES = [
    {"date": "2024-01-01", "value": 412},
    {"date": "2024-01-08", "value": 387},
    {"date": "2024-01-15", "value": 524},
    {"date": "2024-01-22", "value": 461},
    {"date": "2024-01-29", "value": 398},
    {"date": "2024-02-05", "value": 542},
    {"date": "2024-02-12", "value": 475},
    {"date": "2024-02-19", "value": 401},
    {"date": "2024-02-26", "value": 558},
    {"date": "2024-03-04", "value": 489},
    {"date": "2024-03-11", "value": 419},
    {"date": "2024-03-18", "value": 571},
]


# --- translation ----------------------------------------------------------


class StructuredTranslation(BaseModel):
    translated_text: str
    translated_text_iso_code: str
    original_text_iso_code: str


def test_translate_structured(llm) -> None:
    out = llm.with_structured_output(StructuredTranslation, include_raw=True).invoke(
        "Translate the following text into French: 'The UK drinks about 100-160 million cups of tea "
        "every day, and 98% of tea drinkers add milk to their tea.'"
    )
    parsed: StructuredTranslation = out["parsed"]
    assert "fr" in parsed.translated_text_iso_code.lower()
    assert "en" in parsed.original_text_iso_code.lower()
    assert "translate" in precontext_names(out["raw"])


class TargetedTranslation(BaseModel):
    translated_text: str
    target_language: str


def test_translate_es(llm) -> None:
    out = llm.with_structured_output(TargetedTranslation).invoke(
        "Hello, how are you today? I would like to order a coffee. — in Spanish please"
    )
    assert re.search(r"hola|cómo|está|café", out.translated_text.lower())


class SimpleTranslation(BaseModel):
    translated_text: str


def test_translate_long_fr(llm) -> None:
    out = llm.with_structured_output(SimpleTranslation).invoke(
        f'Can you give me this in French? "{LONG_TEXT}"'
    )
    assert len(out.translated_text) >= len(LONG_TEXT) * 0.5
    assert any(
        w in out.translated_text.lower()
        for w in (" le ", " la ", " les ", " des ", " une ", " est ", " pour ", " avec ")
    )


def test_translate_ja(llm) -> None:
    out = llm.with_structured_output(SimpleTranslation).invoke(
        "how do you say 'Thank you for your help' in Japanese?"
    )
    assert re.search(r"[぀-ヿ一-鿿]", out.translated_text)


def test_translate_markup(llm) -> None:
    out = llm.with_structured_output(SimpleTranslation).invoke(
        'Translate this to French, preserving all HTML tags exactly: Click <a href="/start">here</a> '
        "to <b>continue</b>."
    )
    for fragment in ('<a href="/start">', "</a>", "<b>", "</b>"):
        assert fragment in out.translated_text


# --- web search -----------------------------------------------------------


def test_web_search_basic(llm) -> None:
    assert text_of(llm.invoke("Latest news on Nvidia"))


class TeslaFacts(BaseModel):
    founders: list[str]
    year_founded: int
    sources: list[str]


def test_web_search_factual(llm) -> None:
    out = llm.with_structured_output(TeslaFacts).invoke("who founded Tesla and when?")
    assert re.search(r"musk|eberhard|tarpenning", " ".join(out.founders).lower())
    assert out.year_founded == 2003


class Apollo(BaseModel):
    summary: str
    year: int
    month: str
    sources: list[str]


def test_web_search_history(llm) -> None:
    out = llm.with_structured_output(Apollo).invoke("when did Apollo 11 land on the moon? give sources.")
    assert out.year == 1969
    assert "jul" in out.month.lower()
    assert len(out.summary) > 20
    assert out.sources


class NvidiaNews(BaseModel):
    summary: str
    current_stock_price: float
    links: list[str]


def test_web_search_structured(llm) -> None:
    out = llm.with_structured_output(NvidiaNews).invoke("Latest news on Nvidia")
    assert out.summary
    assert out.links


class Person(BaseModel):
    summary: str
    company: str | None
    emails: list[str] | None
    location: str | None


def test_web_search_person(llm) -> None:
    out = llm.with_structured_output(Person, include_raw=True).invoke(
        "Who is Yoeven D Khemlani, his company, his email and where is he based now?"
    )
    assert out["parsed"].summary
    assert precontext_names(out["raw"])


# --- scraping -------------------------------------------------------------


class Listing(BaseModel):
    price: float
    listing_name: str
    seller_name: str
    possible_delivery_time: str


class Listings(BaseModel):
    products: list[Listing]


def test_scrape_ecommerce(llm) -> None:
    out = llm.with_structured_output(Listings, include_raw=True).invoke(
        "get all prices and listing of products for nintendo switch from "
        "https://www.amazon.com/s?k=nintendo+switch+console"
    )
    parsed: Listings = out["parsed"]
    assert parsed.products
    assert parsed.products[0].listing_name
    assert re.search(r"web_extract|search|scraper", " ".join(precontext_names(out["raw"])))


class Post(BaseModel):
    title: str
    points: int


class Posts(BaseModel):
    posts: list[Post]


def test_scrape_hn(llm) -> None:
    out = llm.with_structured_output(Posts).invoke(
        "Extract post titles and points from https://news.ycombinator.com"
    )
    assert out.posts


# --- forecast -------------------------------------------------------------


class Prediction(BaseModel):
    value: float


class Forecast(BaseModel):
    predictions: list[Prediction]


def test_forecast_series(llm) -> None:
    out = llm.with_structured_output(Forecast, include_raw=True).invoke(
        f"Here's our weekly sales for the past 12 weeks: {json.dumps(SERIES)}. "
        "What can we expect over the next month?"
    )
    parsed: Forecast = out["parsed"]
    assert len(parsed.predictions) >= 3
    for p in parsed.predictions:
        assert 0 <= p.value <= 10_000
    assert "forecast" in precontext_names(out["raw"])


# --- code sandbox ---------------------------------------------------------


class Factorial(BaseModel):
    fractional: float


def test_sandbox_factorial(llm) -> None:
    out = llm.with_structured_output(Factorial, include_raw=True).invoke("What is the factorial of 5?")
    assert out["parsed"].fractional == 120
    # The router only reaches for the sandbox when it doesn't already know the answer.
    names = precontext_names(out["raw"])
    if names:
        assert re.search(r"code_execute|code_generation", " ".join(names))


class Answer(BaseModel):
    answer: int


def test_sandbox_count_r(llm) -> None:
    assert llm.with_structured_output(Answer).invoke("How many r's are there in strawberry?").answer == 3


class Script(BaseModel):
    code: str
    sample_input: str
    sample_output: str


def test_sandbox_codegen(llm) -> None:
    out = llm.with_structured_output(Script).invoke(
        "write a python script for getting cpu type using subprocess module and verify your output"
    )
    assert "subprocess" in out.code


# --- function calling -----------------------------------------------------


def test_fc_horoscope(llm) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_horoscope",
                "description": "Get today's horoscope for an astrological sign.",
                "parameters": {
                    "type": "object",
                    "properties": {"sign": {"type": "string"}},
                    "required": ["sign"],
                },
            },
        }
    ]
    bound = llm.bind_tools(tools)
    first: AIMessage = bound.invoke([HumanMessage("Get my horoscope for Taurus")])
    assert first.tool_calls
    assert first.tool_calls[0]["name"] == "get_horoscope"

    second = bound.invoke(
        [
            HumanMessage("Get my horoscope for Taurus"),
            first,
            ToolMessage(
                tool_call_id=first.tool_calls[0]["id"],
                content="Today's horoscope for Taurus: You will have a great day!",
            ),
        ]
    )
    assert text_of(second)
