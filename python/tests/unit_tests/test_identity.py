from __future__ import annotations

import re
from pathlib import Path

import respx
from langchain_core.messages import HumanMessage

from langchain_interfaze import ChatInterfaze, __version__
from tests.unit_tests.conftest import BASIC, mock_json


def test_provider_identity() -> None:
    model = ChatInterfaze(api_key="t")
    assert model._llm_type == "interfaze"
    assert model._get_ls_params()["ls_provider"] == "interfaze"
    assert model.lc_secrets == {"openai_api_key": "INTERFAZE_API_KEY"}
    assert model.get_lc_namespace() == ["langchain_interfaze", "chat_models"]
    assert model.metadata is not None
    assert "langchain-interfaze" in model.metadata["lc_versions"]


def test_version_matches_pyproject() -> None:
    # Read the raw line rather than tomllib, which is 3.11+ and this package is 3.10+.
    pyproject = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
    assert declared is not None, "no version in pyproject.toml"
    assert declared.group(1) == __version__


@respx.mock
def test_model_provider_stamped_on_response() -> None:
    mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    assert model.invoke([HumanMessage("hi")]).response_metadata["model_provider"] == "interfaze"
