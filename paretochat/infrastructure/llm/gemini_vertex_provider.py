"""Gemini model via Vertex AI (GCP credentials required)."""

import os

import google.auth
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI

_DEFAULT_MODEL = "gemini-2.5-pro"


def create_gemini_vertex_model() -> BaseChatModel:
    """Creates a Gemini Chat model backed by Vertex AI.

    Requires the following environment variables:
      GOOGLE_CLOUD_PROJECT   — GCP project ID
      GOOGLE_CLOUD_REGION    — GCP region (default: us-west1)
      GEMINI_VERTEX_MODEL    — model name (default: gemini-2.5-pro)

    See https://reference.langchain.com/python/integrations/langchain_google_genai/ChatGoogleGenerativeAI/
    Available models: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions
    """
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set to use Vertex AI")
    region = os.environ.get("GOOGLE_CLOUD_REGION", "us-west1")
    model = os.environ.get("GEMINI_VERTEX_MODEL", _DEFAULT_MODEL)
    credentials, _ = google.auth.default()
    return ChatGoogleGenerativeAI(
        model=model,
        project=project,
        location=region,
        credentials=credentials,
    )
