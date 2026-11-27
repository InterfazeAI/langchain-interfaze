from __future__ import annotations

import warnings
from collections.abc import Callable

import pytest

from interfaze_langchain import ChatInterfaze, chat_models

KEYLESS_WARNING = (
    "Interfaze SDK is running in keyless mode. For higher rate limits, get an API key at "
    "https://interfaze.ai/dashboard."
)
ENV_KEY = "env-key-1"
EXPLICIT_KEY = "explicit-key-2"


@pytest.fixture(autouse=True)
def _warning_not_yet_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_models, "_keyless_warned", False)


# pytest-socket also warns while a client is built, so only the keyless warnings count.
def keyless_warnings(build: Callable[[], ChatInterfaze]) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        build()
    return [w for w in record if str(w.message) == KEYLESS_WARNING]


def test_first_keyless_client_warns_once_with_the_exact_text() -> None:
    (warning,) = keyless_warnings(ChatInterfaze)
    assert warning.category is UserWarning
    assert warning.filename == __file__


def test_second_keyless_client_does_not_warn() -> None:
    assert len(keyless_warnings(ChatInterfaze)) == 1
    assert keyless_warnings(ChatInterfaze) == []


@pytest.mark.parametrize(
    "build",
    [
        lambda: ChatInterfaze(api_key=EXPLICIT_KEY),
        lambda: ChatInterfaze(openai_api_key=EXPLICIT_KEY),
        lambda: ChatInterfaze(default_headers={"Authorization": f"Bearer {EXPLICIT_KEY}"}),
    ],
    ids=["api_key", "openai_api_key", "authorization header"],
)
def test_client_with_a_key_does_not_warn(build: Callable[[], ChatInterfaze]) -> None:
    assert keyless_warnings(build) == []


def test_client_with_an_env_key_does_not_warn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERFAZE_API_KEY", ENV_KEY)
    assert keyless_warnings(ChatInterfaze) == []


def test_a_keyed_client_does_not_use_up_the_warning() -> None:
    assert keyless_warnings(lambda: ChatInterfaze(api_key=EXPLICIT_KEY)) == []
    assert len(keyless_warnings(ChatInterfaze)) == 1
