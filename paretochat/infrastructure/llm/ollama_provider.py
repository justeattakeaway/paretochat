"""Ollama local LLM provider (no API key required)."""

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

_DEFAULT_MODEL = "llama3.1:8b"
_DEFAULT_BASE_URL = "http://localhost:11434"

# Models confirmed to support both structured output AND tool calling (required by
# McdmExpertAgent and SessionSummarizerAgent which use LangChain tools):
#   qwen2.5:7b (recommended), qwen2.5:14b, llama3.1:8b, llama3.3:70b, mistral-nemo
# Avoid:
#   llama3.2 (3B — too small to reliably follow complex schemas)
#   llama2, phi3, gemma2, codellama (no tool calling)
#   deepseek-r1:* (thinking tokens break structured output parsing)


def create_ollama_model() -> BaseChatModel:
    """Creates an Ollama Chat model backed by a local Ollama server.

    Uses Ollama's native JSON schema constrained generation for structured output
    (grammar-based, enforces the schema at the token level — much more reliable
    than the default function_calling method).

    Requires:
      - langchain-ollama installed: uv sync --extra ollama
      - Ollama running (https://ollama.com/download) with a tool-calling model:
          ollama pull qwen2.5:7b

    Environment variables:
      - OLLAMA_MODEL    — model name (default: llama3.1:8b)
      - OLLAMA_BASE_URL — server URL (default: http://localhost:11434)
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise ImportError("langchain-ollama is not installed. Install it with: uv sync --extra ollama") from exc

    class _ChatOllama(ChatOllama):
        # ChatOllama defaults with_structured_output to method="function_calling",
        # which relies on the model following tool call format and is unreliable.
        # method="json_schema" uses Ollama's grammar-constrained generation instead,
        # enforcing the schema at the token level regardless of model size.
        def with_structured_output(self, schema: Any, *, method: str = "json_schema", **kwargs: Any) -> Any:
            return super().with_structured_output(schema, method=method, **kwargs)

    model_name = os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)
    base_url = os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
    return _ChatOllama(model=model_name, base_url=base_url)
