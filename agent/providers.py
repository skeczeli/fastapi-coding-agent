"""Provider configuration — the one place that picks the LLM/embedding backend.

Both the chat path (``llm.complete``) and the embedding path (``rag.store``) build
their OpenAI-SDK client from here, so switching providers — OpenAI ↔ Gemini ↔ any
OpenAI-compatible endpoint — is **env-only**, no code change. This is the seam that
lets us run free on Gemini now and flip back to OpenAI when credits are topped up.

Chat and embeddings are configured independently, because they can live on
different backends (e.g. Gemini chat + OpenAI embeddings):

    Chat        AGENT_LLM_BASE_URL    AGENT_LLM_API_KEY    AGENT_LLM_MODEL
                AGENT_LLM_MAX_TOKENS_PARAM   (max_completion_tokens | max_tokens)
    Embeddings  AGENT_EMBED_BASE_URL  AGENT_EMBED_API_KEY  AGENT_EMBED_MODEL

Anything unset falls back to OpenAI defaults (the project's original behaviour),
with ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` honoured as a final fallback so an
OpenAI-only setup needs no AGENT_* vars at all.

Note: switching the *embedding* provider changes the vector space (and dimensions),
so the RAG store must be re-ingested (``python -m agent.rag.ingest --rebuild``)
after changing ``AGENT_EMBED_*``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# OpenAI defaults — used when no AGENT_* override is set.
_DEFAULT_CHAT_MODEL = "gpt-5-nano"
_DEFAULT_EMBED_MODEL = "text-embedding-3-small"
# OpenAI's chat endpoint wants ``max_completion_tokens``; most OpenAI-compatible
# layers (Gemini, Groq, ...) want the older ``max_tokens``. Switchable per provider.
_DEFAULT_MAX_TOKENS_PARAM = "max_completion_tokens"


@dataclass(frozen=True)
class ProviderConfig:
    """Everything needed to point the OpenAI SDK at one backend.

    Attributes:
        model: Model id to request.
        base_url: Endpoint override, or ``None`` to use the SDK's OpenAI default.
        api_key: API key, or ``None`` to let the SDK read ``OPENAI_API_KEY``.
        max_tokens_param: Name of the output-cap kwarg (chat only; ignored for
            embeddings). ``max_completion_tokens`` for OpenAI, ``max_tokens`` else.
    """

    model: str
    base_url: str | None = None
    api_key: str | None = None
    max_tokens_param: str = _DEFAULT_MAX_TOKENS_PARAM

    def client_kwargs(self) -> dict[str, str]:
        """Kwargs for ``OpenAI(**kwargs)`` — only the ones explicitly set.

        Omitting ``base_url``/``api_key`` when ``None`` lets the SDK fall back to
        its own defaults (``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``).
        """
        kwargs: dict[str, str] = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.api_key:
            kwargs["api_key"] = self.api_key
        return kwargs


def _first_env(*names: str) -> str | None:
    """Return the first set, non-empty environment variable among ``names``."""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def chat_config() -> ProviderConfig:
    """Resolve the chat provider from the environment (OpenAI by default)."""
    return ProviderConfig(
        model=os.getenv("AGENT_LLM_MODEL") or _DEFAULT_CHAT_MODEL,
        base_url=_first_env("AGENT_LLM_BASE_URL", "OPENAI_BASE_URL"),
        api_key=_first_env("AGENT_LLM_API_KEY", "OPENAI_API_KEY"),
        max_tokens_param=os.getenv("AGENT_LLM_MAX_TOKENS_PARAM") or _DEFAULT_MAX_TOKENS_PARAM,
    )


def embed_config() -> ProviderConfig:
    """Resolve the embedding provider from the environment (OpenAI by default)."""
    return ProviderConfig(
        model=os.getenv("AGENT_EMBED_MODEL") or _DEFAULT_EMBED_MODEL,
        base_url=_first_env("AGENT_EMBED_BASE_URL", "OPENAI_BASE_URL"),
        api_key=_first_env("AGENT_EMBED_API_KEY", "OPENAI_API_KEY"),
    )
