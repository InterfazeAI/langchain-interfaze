from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from langchain_interfaze import ChatInterfaze

HAS_KEY = bool(os.environ.get("INTERFAZE_API_KEY"))
BASE_URL = os.environ.get("INTERFAZE_BASE_URL")

requires_key = pytest.mark.skipif(not HAS_KEY, reason="INTERFAZE_API_KEY is not set")

# Live calls that run internal tools (OCR / STT / scrape / forecast) are slow.
SLOW = 300
FAST = 120

FIXTURES = Path(os.environ.get("INTERFAZE_FIXTURES", Path.home() / "interfaze-sdk-tests" / "fixtures"))

IMAGES = {
    "receipt": "https://jigsawstack.com/preview/vocr-example.jpg",
    "receipt_items": "https://cdn.hashnode.com/res/hashnode/image/upload/v1741819852493/10b20478-03da-4ed9-be86-0dc33e97a673.jpeg?auto=compress,format&format=webp",
    "id_medium": "https://miro.medium.com/v2/resize:fit:698/1*q_FimDPBNMvJXJyDtXT3Jg.jpeg",
    "id_jpg": "https://r2public.jigsawstack.com/interfaze/examples/id.jpg",
    "multilang": "https://cdn.hashnode.com/res/hashnode/image/upload/v1746576859594/31e54f33-e825-4930-8fe3-8a1380ba9e16.jpeg?auto=compress,format&format=webp",
    "katana": "https://jigsawstack.com/preview/object-detection-example-input.jpg",
    "bus": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
    "gui_form": "https://r2public.jigsawstack.com/interfaze/examples/GUI_form.png",
    "construction": "https://r2public.jigsawstack.com/interfaze/examples/construction.png",
    "gore": "https://plus.unsplash.com/premium_photo-1695691596554-1a07f2a8cf34?q=80&w=1587&auto=format&fit=crop",
    "missing": "https://jigsawstack.com/preview/this-image-definitely-does-not-exist-xyz123.jpg",
}

FILES = {
    "attention_pdf": "https://arxiv.org/pdf/1706.03762",
    "stt_short": "https://r2public.jigsawstack.com/interfaze/examples/stt_medical_short.mp4",
    "stt_multi": "https://r2public.jigsawstack.com/interfaze/examples/stt_multispeaker.mp3",
    "stt_call": "https://r2public.jigsawstack.com/interfaze/examples/stt_call.mp3",
    "video": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/360/Big_Buck_Bunny_360_10s_1MB.mp4",
}


def chat(**kwargs: Any) -> ChatInterfaze:
    if BASE_URL:
        kwargs.setdefault("base_url", BASE_URL)
    kwargs.setdefault("max_retries", 1)
    return ChatInterfaze(**kwargs)


def fresh_chat(**kwargs: Any) -> ChatInterfaze:
    """Fresh model output — the semantic cache otherwise replays a prior answer."""
    return chat(bypass_cache=True, **kwargs)


def receipt_b64() -> str:
    """base64 JPEG receipt — GT: "The Marco Polo Kitch" / 15.15 / 2018-05-06."""
    return (FIXTURES / "receipt.b64").read_text().strip()


def image_part(url: str) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": url}}


def file_part(url: str, filename: str | None = None) -> dict[str, Any]:
    file: dict[str, Any] = {"file_data": url}
    if filename:
        file["filename"] = filename
    return {"type": "file", "file": file}


def video_part(url: str) -> dict[str, Any]:
    return {"type": "video", "url": url}


def precontext_names(message: Any) -> list[str]:
    """Names of the internal tools Interfaze ran, from `response_metadata.precontext`."""
    entries = message.response_metadata.get("precontext") or []
    return [e.get("name") for e in entries if isinstance(e, dict) and e.get("name")]


def text_of(message: Any) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)
