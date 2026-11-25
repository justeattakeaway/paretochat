"""Dispatch LLM model creation to the configured provider."""

import logging
import os
import threading

from langchain_core.language_models.chat_models import BaseChatModel

_logger = logging.getLogger(__name__)

_SUPPORTED_PROVIDERS = ("gemini-vertex", "gemini-api", "ollama", "claude-cli")

_ACCESS_CHECK_PROMPT = "Reply with the single word OK."

# (provider, model name) combinations that already passed the access check in this process.
_verified: set[tuple[str, str]] = set()
_verified_lock = threading.Lock()


class LLMAccessError(RuntimeError):
    """The configured LLM provider or model is not reachable with the current credentials."""


def create_llm_model() -> BaseChatModel:
    """Create an LLM model for the configured provider, after verifying once that it can be reached.

    Every code path that needs an LLM goes through this function, so the access check runs here
    (once per process and provider/model) and does not have to be repeated by callers.
    See `ensure_llm_access` for the check itself.
    """
    model = _build_llm_model()
    _verify_access(model)
    return model


def ensure_llm_access() -> None:
    """Verify early that the configured LLM can be used; no-op after the first success in this process.

    Call this at the start of a session to fail fast with a clear message instead of failing in the middle
    of the first agent call. `create_llm_model` runs the same check, so calling this is optional.

    Raises:
        LLMAccessError: if the model cannot be created or does not return a usable answer.
    """
    create_llm_model()


def _provider() -> str:
    return os.environ.get("LLM_PROVIDER", "gemini-vertex").strip().lower()


def _model_name(model: BaseChatModel) -> str:
    return str(getattr(model, "model", None) or getattr(model, "model_name", None) or type(model).__name__)


def _verify_access(model: BaseChatModel) -> None:
    """Send one tiny plain-text request to the model. Cached per (provider, model) after success."""
    provider, name = _provider(), _model_name(model)
    key = (provider, name)
    with _verified_lock:
        if key in _verified:
            return
        _logger.info("Checking access to LLM provider %s (model %s)", provider, name)
        try:
            _provider_preflight(provider)
        except Exception as e:
            raise LLMAccessError(_access_error_message(provider, name, str(e))) from e
        try:
            response = model.invoke(_ACCESS_CHECK_PROMPT)
        except Exception as e:
            raise LLMAccessError(_access_error_message(provider, name, f"{type(e).__name__}: {e}")) from e
        if not response.text.strip():
            raise LLMAccessError(_access_error_message(provider, name, "the model returned an empty response"))
        _verified.add(key)


def _provider_preflight(provider: str) -> None:
    """Provider-specific checks that a plain test request cannot reveal."""
    if provider == "claude-cli":
        from paretochat.infrastructure.llm.claude_cli_provider import check_claude_login

        check_claude_login()


def _access_error_message(provider: str, model_name: str, detail: str) -> str:
    return (
        f"Cannot use LLM provider {provider!r} (model {model_name!r}): {detail}\n"
        "Check your login or credentials and that the model is available to you. "
        "Set LLM_PROVIDER and the provider's model variable to change them; "
        "see the 'LLM Configuration' section of the README."
    )


def _build_llm_model() -> BaseChatModel:
    """Create an LLM model for the configured provider.

    Provider is selected via the LLM_PROVIDER environment variable (default: gemini-vertex).

    Supported values:
      - gemini-vertex  (default) — Gemini via Vertex AI; requires GCP credentials
      - gemini-api               — Gemini via direct API key; requires GOOGLE_API_KEY
      - ollama                   — Local Ollama server; requires langchain-ollama extra
    """
    provider = _provider()

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
