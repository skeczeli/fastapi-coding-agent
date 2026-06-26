"""Provider-selection tests — env drives the backend, OpenAI is the default."""

from __future__ import annotations

import pytest

from agent import providers


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    """Start each test from a clean slate (no AGENT_*/OPENAI_* leaking in)."""
    for var in (
        "AGENT_LLM_BASE_URL",
        "AGENT_LLM_API_KEY",
        "AGENT_LLM_MODEL",
        "AGENT_LLM_MAX_TOKENS_PARAM",
        "AGENT_EMBED_BASE_URL",
        "AGENT_EMBED_API_KEY",
        "AGENT_EMBED_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


def test_chat_defaults_to_openai_when_unset():
    cfg = providers.chat_config()
    assert cfg.model == "gpt-5-nano"
    assert cfg.base_url is None
    assert cfg.max_tokens_param == "max_completion_tokens"
    # No base_url/api_key -> SDK falls back to its own OpenAI defaults.
    assert cfg.client_kwargs() == {}


def test_chat_reads_gemini_style_overrides(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    monkeypatch.setenv("AGENT_LLM_API_KEY", "gem-key")
    monkeypatch.setenv("AGENT_LLM_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("AGENT_LLM_MAX_TOKENS_PARAM", "max_tokens")

    cfg = providers.chat_config()
    assert cfg.model == "gemini-2.0-flash"
    assert cfg.max_tokens_param == "max_tokens"
    assert cfg.client_kwargs() == {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key": "gem-key",
    }


def test_openai_key_is_the_final_fallback(monkeypatch):
    # With no AGENT_LLM_API_KEY, the plain OPENAI_API_KEY is still honoured.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert providers.chat_config().client_kwargs() == {"api_key": "sk-openai"}


def test_agent_key_takes_precedence_over_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("AGENT_LLM_API_KEY", "agent-key")
    assert providers.chat_config().api_key == "agent-key"


def test_embed_config_is_independent_from_chat(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_MODEL", "gemini-2.0-flash")
    monkeypatch.setenv("AGENT_EMBED_MODEL", "text-embedding-004")
    monkeypatch.setenv("AGENT_EMBED_API_KEY", "embed-key")

    embed = providers.embed_config()
    assert embed.model == "text-embedding-004"
    assert embed.client_kwargs() == {"api_key": "embed-key"}
    # Chat model didn't bleed into the embedding config.
    assert providers.chat_config().model == "gemini-2.0-flash"


def test_empty_env_var_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("AGENT_LLM_MODEL", "")
    assert providers.chat_config().model == "gpt-5-nano"
