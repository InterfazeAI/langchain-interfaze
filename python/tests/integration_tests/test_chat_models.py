from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_interfaze import ChatInterfaze


class TestChatInterfazeIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[ChatInterfaze]:
        return ChatInterfaze

    @property
    def chat_model_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {"model": "interfaze-beta"}
        base_url = os.environ.get("INTERFAZE_BASE_URL")
        if base_url:
            params["base_url"] = base_url
        return params

    @property
    def has_tool_calling(self) -> bool:
        return True

    @property
    def has_tool_choice(self) -> bool:
        # Interfaze accepts `tools` but drops `tool_choice`; the router always decides.
        return False

    @property
    def has_structured_output(self) -> bool:
        return True

    @property
    def supports_json_mode(self) -> bool:
        return True

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_image_urls(self) -> bool:
        return True

    @property
    def supports_pdf_inputs(self) -> bool:
        return True

    @property
    def supports_audio_inputs(self) -> bool:
        return True

    @property
    def supports_video_inputs(self) -> bool:
        return True

    @property
    def supports_image_tool_message(self) -> bool:
        # Interfaze rejects assistant/tool messages carrying image content blocks.
        return False

    @property
    def supports_pdf_tool_message(self) -> bool:
        return False

    @property
    def supports_anthropic_inputs(self) -> bool:
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        return True

    @pytest.mark.xfail(
        reason="Interfaze rejects assistant messages whose content is a list of blocks "
        "(400 invalid_request on messages.N); only string content is accepted there."
    )
    def test_tool_message_histories_list_content(self, *args: Any) -> None:
        super().test_tool_message_histories_list_content(*args)

    @pytest.mark.xfail(
        reason="Interfaze drops `tool_choice` and routes tool use itself, so a user tool "
        "the model can answer without (here: the weather) is not reliably called."
    )
    def test_agent_loop(self, model: BaseChatModel) -> None:
        super().test_agent_loop(model)
