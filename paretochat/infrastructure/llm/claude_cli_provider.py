# Claude models via Claude CLI through Oauth

import json
import os
from typing import Any, Callable, Mapping, Sequence, Union

from langchain_claude_cli import ChatClaudeCli
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool

_DEFAULT_MODEL = "claude-sonnet-4-5"


class _ChatClaudeCli(ChatClaudeCli):
    """ChatClaudeCli subclass that advertises native structured output support.

    `create_agent` from langchain checks `model.profile["structured_output"]` to
    decide between ToolStrategy (tool_choice='any', CLI cannot enforce) and
    ProviderStrategy (native JSON schema). This subclass opts into ProviderStrategy
    and translates the OpenAI-style `response_format` kwarg that ProviderStrategy
    injects into bind_tools() into the CLI-native `output_format` kwarg.
    It also wraps the result so that `additional_kwargs["structured_output"]` is
    serialized into `content`, which is where ProviderStrategyBinding.parse() looks.
    """

    def _resolve_model_profile(self) -> dict[str, Any]:
        return {"structured_output": True}

    def bind_tools(
        self,
        tools: Sequence[Union[Mapping[str, Any], type, Callable, BaseTool]],
        *,
        tool_choice: dict[str, str] | str | None = None,
        **kwargs: Any,
    ):
        # ProviderStrategy passes response_format={"type": "json_schema",
        # "json_schema": {"name": ..., "schema": {...}}} (OpenAI format).
        # The CLI runner only understands output_format={"type": "json_schema",
        # "schema": {...}}, so we convert and also wrap to expose the parsed
        # structured output in content for ProviderStrategyBinding.parse().
        response_format = kwargs.pop("response_format", None)
        if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            json_schema_spec = response_format.get("json_schema", {})
            kwargs["output_format"] = {
                "type": "json_schema",
                "schema": json_schema_spec.get("schema", json_schema_spec),
            }
            bound = super().bind_tools(tools, tool_choice=tool_choice, **kwargs)
            return bound | RunnableLambda(_expose_structured_output_in_content)

        return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)


def _expose_structured_output_in_content(msg: Any) -> Any:
    """Ensure message.content contains valid JSON for ProviderStrategyBinding.parse().

    ProviderStrategyBinding.parse() reads message.content via json.loads().
    ChatClaudeCli puts the CLI's native structured output in
    additional_kwargs["structured_output"] (a parsed dict), not in content.
    When that key is present we serialize it back to a JSON string in content
    so the generic LangChain parsing path can find it.
    If additional_kwargs["structured_output"] is absent, the content is left
    as-is (it may already be a JSON string when the model emitted it as text).
    """
    if not isinstance(msg, AIMessage):
        return msg
    structured = msg.additional_kwargs.get("structured_output")
    if structured is None:
        return msg
    return AIMessage(
        content=json.dumps(structured),
        tool_calls=msg.tool_calls,
        usage_metadata=msg.usage_metadata,
        response_metadata=msg.response_metadata,
        additional_kwargs=msg.additional_kwargs,
        id=msg.id,
    )


def create_claude_model() -> BaseChatModel:
    """Create Claude Chat model using Claude CLI oauth"""
    model = os.environ.get("CLAUDE_MODEL", _DEFAULT_MODEL)
    return _ChatClaudeCli(model=model)
