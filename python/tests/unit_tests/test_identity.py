from __future__ import annotations

from pathlib import Path

import respx
import tomllib
from langchain_core.messages import HumanMessage

from langchain_interfaze import ChatInterfaze, __version__
from tests.unit_tests.conftest import BASIC, mock_json


def test_provider_identity() -> None:
    model = ChatInterfaze(api_key="t")
    assert model._llm_type == "interfaze-beta"
    assert model._get_ls_params()["ls_provider"] == "interfaze"
    assert model.lc_secrets == {"openai_api_key": "INTERFAZE_API_KEY"}
    assert model.get_lc_namespace() == ["langchain_interfaze", "chat_models"]
    assert model.metadata is not None
    assert "langchain-interfaze" in model.metadata["lc_versions"]


def test_version_matches_pyproject() -> None:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    assert tomllib.loads(pyproject.read_text())["project"]["version"] == __version__


@respx.mock
def test_model_provider_stamped_on_response() -> None:
    mock_json(BASIC)
    model = ChatInterfaze(api_key="t")
    assert model.invoke([HumanMessage("hi")]).response_metadata["model_provider"] == "interfaze"
