from __future__ import annotations

from typing import Any

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_interfaze import ChatInterfaze


class TestChatInterfazeUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatInterfaze]:
        return ChatInterfaze

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {"api_key": "test", "model": "interfaze-beta"}
