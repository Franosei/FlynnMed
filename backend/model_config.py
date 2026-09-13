"""Single source of truth for the application's default OpenAI model."""

from __future__ import annotations

import os


DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"


def configured_openai_model() -> str:
    """Return the configured application model, falling back to the supported default."""
    return os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_OPENAI_MODEL
