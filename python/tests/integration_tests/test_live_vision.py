from __future__ import annotations

from typing import Any, Literal

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from tests.integration_tests.conftest import (
    FILES,
    IMAGES,
    SLOW,
    chat,
    file_part,
    image_part,
    precontext_names,
    receipt_b64,
    requires_key,
)

pytestmark = [requires_key, pytest.mark.timeout(SLOW)]


@pytest.fixture(scope="module")
def llm():
    return chat()


def ask(prompt: str, part: dict[str, Any]) -> HumanMessage:
    return HumanMessage(content=[{"type": "text", "text": prompt}, part])


class Box(BaseModel):
    top_left_x: float
    top_left_y: float
    bottom_right_x: float
    bottom_right_y: float


# --- ocr ------------------------------------------------------------------


class IdDocument(BaseModel):
    full_first_name: str
    full_last_name: str
    full_address: str | None
    email: str | None
    id_type: str


def test_ocr_id_document(llm) -> None:
    out = llm.with_structured_output(IdDocument, include_raw=True).invoke(
        [ask("Extract information from the image based on the schema.", image_part(IMAGES["id_medium"]))]
    )
    parsed: IdDocument = out["parsed"]
    assert "iv" in parsed.full_first_name.lower()
    assert parsed.full_last_name.lower().replace("ñ", "n").find("munoz") >= 0
    assert parsed.id_type
    assert "ocr" in precontext_names(out["raw"])


class DriverLicence(BaseModel):
    first_name: str
    last_name: str
    dob: str
    driver_licence_number: str


def test_ocr_id_jpg(llm) -> None:
    out = llm.with_structured_output(DriverLicence, include_raw=True).invoke(
        [ask("Extract the details from this ID", image_part(IMAGES["id_jpg"]))]
    )
    parsed: DriverLicence = out["parsed"]
    assert parsed.first_name and parsed.dob and parsed.driver_licence_number
    assert "ocr" in precontext_names(out["raw"])


class LineItem(BaseModel):
    name: str
    price: str


class ReceiptFields(BaseModel):
    items: list[LineItem]
    highlighted_items: list[LineItem]
    total_cost: str
    tax: str


def test_ocr_receipt_fields(llm) -> None:
    out = llm.with_structured_output(ReceiptFields, include_raw=True).invoke(
        [ask("Extract text from the image.", image_part(IMAGES["receipt_items"]))]
    )
    parsed: ReceiptFields = out["parsed"]
    assert parsed.items
    assert "144.02" in parsed.total_cost
    assert "4.58" in parsed.tax
    assert "gale" in " ".join(i.name for i in parsed.highlighted_items).lower()
    assert "ocr" in precontext_names(out["raw"])


class QueryResult(BaseModel):
    text: str
    confidence: float


class StoreLocation(BaseModel):
    query_results: list[QueryResult]
    text: str
    confidence: float


def test_ocr_store_location(llm) -> None:
    out = llm.with_structured_output(StoreLocation).invoke(
        [ask("Where is this store located?", image_part(IMAGES["receipt"]))]
    )
    haystack = f"{' '.join(r.text for r in out.query_results)} {out.text}".lower()
    assert "greenwood" in haystack


class Word(Box):
    text: str


class WordBoxes(BaseModel):
    text: str
    words: list[Word]


def test_ocr_word_bboxes(llm) -> None:
    out = llm.with_structured_output(WordBoxes, include_raw=True).invoke(
        [ask("extract every word and its position from this receipt", image_part(IMAGES["receipt"]))]
    )
    parsed: WordBoxes = out["parsed"]
    assert len(parsed.words) >= 10
    for w in parsed.words:
        assert w.top_left_x >= 0
        assert w.top_left_y >= 0
        assert w.bottom_right_x >= w.top_left_x - 2
        assert w.bottom_right_y >= w.top_left_y - 2
    assert "ocr" in precontext_names(out["raw"])


class Paper(BaseModel):
    title: str
    authors: list[str]


def test_ocr_pdf_title_authors(llm) -> None:
    out = llm.with_structured_output(Paper, include_raw=True).invoke(
        [
            ask(
                "Extract the title and author names from the first page.",
                file_part(FILES["attention_pdf"], "attention.pdf"),
            )
        ]
    )
    parsed: Paper = out["parsed"]
    assert "attention" in parsed.title.lower()
    assert "vaswani" in " ".join(parsed.authors).lower()
    assert "ocr" in precontext_names(out["raw"])


class Translation(BaseModel):
    text_in_original_language: str
    text_in_telugu: str
    width_of_image: float
    height_of_image: float


class Multilang(BaseModel):
    translations: list[Translation]


def test_ocr_multilang(llm) -> None:
    out = llm.with_structured_output(Multilang).invoke(
        [ask("Extract information from the image based on the schema.", image_part(IMAGES["multilang"]))]
    )
    assert out.translations


class LayoutElement(Box):
    type: Literal["heading", "paragraph", "formula", "figure", "table", "caption", "list"]
    content: str


class Page(BaseModel):
    page_number: int
    elements: list[LayoutElement]


class Layout(BaseModel):
    pages: list[Page]


def test_ocr_document_layout(llm) -> None:
    out = llm.with_structured_output(Layout, include_raw=True).invoke(
        [
            ask(
                "extract the layout elements (headings, paragraphs, figures, tables) with bounding "
                "boxes from the first page",
                file_part(FILES["attention_pdf"], "attention.pdf"),
            )
        ]
    )
    parsed: Layout = out["parsed"]
    assert parsed.pages
    elements = [e for p in parsed.pages for e in p.elements]
    assert elements
    assert any(e.type in {"heading", "paragraph"} for e in elements)
    assert "ocr" in precontext_names(out["raw"])


# --- document extraction / markdown ---------------------------------------


def inline_receipt() -> dict[str, Any]:
    return image_part(f"data:image/jpeg;base64,{receipt_b64()}")


class BillItem(BaseModel):
    description: str
    price: float | None


class Bill(BaseModel):
    vendor_name: str
    total_amount: float
    bill_date: str
    line_items: list[BillItem]


def test_doc_invoice_exact(llm) -> None:
    out = llm.with_structured_output(Bill).invoke(
        [
            ask(
                "Extract the bill data from this receipt: vendor_name, total_amount (number), "
                "bill_date (YYYY-MM-DD), and line_items[{description, price}].",
                inline_receipt(),
            )
        ]
    )
    assert abs(out.total_amount - 15.15) <= 0.01
    assert out.bill_date == "2018-05-06"
    assert "marco polo" in out.vendor_name.lower()
    assert out.line_items


class Markdown(BaseModel):
    markdown: str


def test_md_image_to_md(llm) -> None:
    out = llm.with_structured_output(Markdown).invoke(
        [ask("Convert this receipt image to markdown.", inline_receipt())]
    )
    assert len(out.markdown) >= 80
    assert "marco polo" in out.markdown.lower()
    assert "mocha" in out.markdown.lower()
    assert "15.15" in out.markdown


def test_md_pdf_to_md(llm) -> None:
    import re

    out = llm.with_structured_output(Markdown, include_raw=True).invoke(
        [
            ask(
                "Convert the first page of this document to markdown.",
                file_part(FILES["attention_pdf"], "attention.pdf"),
            )
        ]
    )
    # Heading style is stable; bold emphasis is not, so it is not asserted.
    md: str = out["parsed"].markdown
    assert re.search(r"#{1,3}\s*attention is all you need", md, re.IGNORECASE)
    assert len(md) > 200
    assert "ocr" in precontext_names(out["raw"])


# --- object / gui detection -----------------------------------------------


class NamedObject(BaseModel):
    name: str


class Objects(BaseModel):
    objects: list[NamedObject]


def test_object_detection_absent(llm) -> None:
    out = llm.with_structured_output(Objects).invoke(
        [ask("detect elephants in this image", image_part(IMAGES["katana"]))]
    )
    assert out.objects == []


class BoxedObject(Box):
    name: str


class BoxedObjects(BaseModel):
    objects: list[BoxedObject]


def test_object_detection_bbox(llm) -> None:
    out = llm.with_structured_output(BoxedObjects, include_raw=True).invoke(
        [ask("detect the position of the katana in this image", image_part(IMAGES["katana"]))]
    )
    parsed: BoxedObjects = out["parsed"]
    assert parsed.objects
    box = parsed.objects[0]
    assert "katana" in box.name.lower()
    assert abs(box.top_left_x - 1078) <= 15
    assert abs(box.top_left_y - 474) <= 15
    assert abs(box.bottom_right_x - 1188) <= 15
    assert abs(box.bottom_right_y - 1026) <= 15
    assert "object_detection" in precontext_names(out["raw"])


def test_object_detection_multi(llm) -> None:
    out = llm.with_structured_output(BoxedObjects, include_raw=True).invoke(
        [ask("detect all objects with bounding boxes", image_part(IMAGES["bus"]))]
    )
    # Recall on this image swings from 1 to 8 objects run to run, and the labels vary
    # ("bus" vs a bare "object") — reproduced identically through the core interfaze SDK.
    # What the integration owns is the round-trip: well-formed boxes and the precontext.
    parsed: BoxedObjects = out["parsed"]
    assert parsed.objects
    for o in parsed.objects:
        assert o.bottom_right_x >= o.top_left_x
        assert o.bottom_right_y >= o.top_left_y
    assert "object_detection" in precontext_names(out["raw"])


class GuiElement(Box):
    name: str


class Gui(BaseModel):
    gui_elements: list[GuiElement]


def test_gui_detection_form(llm) -> None:
    out = llm.with_structured_output(Gui, include_raw=True).invoke(
        [ask("find all the text fields and the clear form button", image_part(IMAGES["gui_form"]))]
    )
    # Element count swings between 1 and 12 across runs (reproduced with the core
    # interfaze SDK), so assert the envelope rather than a per-field inventory.
    parsed: Gui = out["parsed"]
    assert len(parsed.gui_elements) >= 1
    for e in parsed.gui_elements:
        assert e.bottom_right_x <= 3600
        assert e.bottom_right_y <= 2338
    assert "gui_detection" in precontext_names(out["raw"])


class BoxedText(Box):
    text: str


class ObjectsAndText(BaseModel):
    objects: list[BoxedObject]
    texts: list[BoxedText]


def test_object_detection_with_text(llm) -> None:
    out = llm.with_structured_output(ObjectsAndText, include_raw=True).invoke(
        [ask("Get the position of the crane in the image and any text", image_part(IMAGES["construction"]))]
    )
    # The crane is not always detected; what must hold is that detection + OCR ran and
    # the schema came back well-formed.
    assert isinstance(out["parsed"].objects, list)
    assert precontext_names(out["raw"])
