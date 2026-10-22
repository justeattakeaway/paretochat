"""LLM model factory — delegates to the provider selected via LLM_PROVIDER env var."""

from langchain_core.language_models.chat_models import BaseChatModel

from paretochat.infrastructure.llm.provider_factory import create_llm_model as _create


def create_llm_model() -> BaseChatModel:
    """Creates an LLM model for the configured provider.

    Provider is controlled by the LLM_PROVIDER environment variable.
    Defaults to 'gemini-vertex' (Vertex AI, requires GCP credentials).
    See provider_factory.py for all supported values.
    """
    return _create()
