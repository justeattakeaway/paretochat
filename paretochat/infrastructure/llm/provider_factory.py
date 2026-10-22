"""Dispatch LLM model creation to the configured provider."""

import os

from langchain_core.language_models.chat_models import BaseChatModel

_SUPPORTED_PROVIDERS = ("gemini-vertex", "gemini-api", "ollama", "claude-cli")


def create_llm_model() -> BaseChatModel:
    """Create an LLM model for the configured provider.

    Provider is selected via the LLM_PROVIDER environment variable (default: gemini-vertex).

    Supported values:
      - gemini-vertex  (default) — Gemini via Vertex AI; requires GCP credentials
      - gemini-api               — Gemini via direct API key; requires GOOGLE_API_KEY
      - ollama                   — Local Ollama server; requires langchain-ollama extra
    """
    provider = os.environ.get("LLM_PROVIDER", "gemini-vertex").strip().lower()

    if provider == "gemini-vertex":
        from paretochat.infrastructure.llm.gemini_vertex_provider import create_gemini_vertex_model

        return create_gemini_vertex_model()

    if provider == "gemini-api":
        from paretochat.infrastructure.llm.gemini_api_provider import create_gemini_api_model

        return create_gemini_api_model()

    if provider == "ollama":
        from paretochat.infrastructure.llm.ollama_provider import create_ollama_model

        return create_ollama_model()

    if provider == "claude-cli":
        from paretochat.infrastructure.llm.claude_cli_provider import create_claude_model

        return create_claude_model()

    raise ValueError(f"Unknown LLM_PROVIDER={provider!r}. Supported values: {', '.join(_SUPPORTED_PROVIDERS)}")
