"""Gemini model via Google AI Studio (direct API key, no GCP required)."""

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

_DEFAULT_MODEL = "gemini-2.0-flash"


def create_gemini_api_model() -> BaseChatModel:
    """Creates a Gemini Chat model using the Google AI Studio API.

    Requires GOOGLE_API_KEY to be set.
    Get a free key at https://aistudio.google.com/apikey

    Environment variables:
      - GOOGLE_API_KEY   — required
      - GEMINI_API_MODEL — model name (default: gemini-2.0-flash)
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set. Get a free API key at https://aistudio.google.com/apikey")
    model = os.environ.get("GEMINI_API_MODEL", _DEFAULT_MODEL)
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
