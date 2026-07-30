from __future__ import annotations

from typing import Any

from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_interfaze import ChatInterfaze


class TestChatInterfazeIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[ChatInterfaze]:
        return ChatInterfaze

    @property
    def chat_model_params(self) -> dict[str, Any]:
        return {"model": "interfaze-beta"}
