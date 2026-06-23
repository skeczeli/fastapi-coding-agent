"""The single place that calls the LLM.

Everything that needs the model goes through ``complete()`` — the harness, the
orchestrator, and every subagent. Centralizing it means observability (#B4) can
wrap one function, and tests can run offline via the mock mode.

Mock mode: set ``AGENT_LLM_MOCK=1`` to return canned responses without hitting
the OpenAI API. Other devs rely on this to test their lanes without a key/cost.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

# Default model. gpt-5-nano needs a high max_completion_tokens because it spends
# internal reasoning tokens — too low and it returns empty content (TP1 lesson).
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_MAX_COMPLETION_TOKENS = 4000


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized result of one LLM call.

    Attributes:
        content: The assistant text (may be empty when only tool calls are made).
        tool_calls: Tools the model wants executed before continuing.
        raw: The provider's raw response object, for debugging/observability.
    """

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


_client = None


def _get_client():
    """Lazily build the OpenAI client so importing this module needs no API key."""
    global _client
    if _client is None:
        from openai import OpenAI  # imported lazily; core import stays light

        _client = OpenAI()
    return _client


def _mock_response(messages: list[dict], tools: list[dict] | None) -> LLMResponse:
    """Deterministic stand-in used when AGENT_LLM_MOCK is set."""
    last = messages[-1]["content"] if messages else ""
    return LLMResponse(content=f"[mock] received: {last}", tool_calls=[], raw=None)


def complete(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> LLMResponse:
    """Call the LLM once and return a normalized ``LLMResponse``.

    Args:
        messages: OpenAI-style chat messages.
        tools: Tool schemas in OpenAI tool-calling format (see ``tools.schemas``).
        model: Model id (defaults to gpt-5-nano).
        max_completion_tokens: Output cap; kept high for reasoning models.

    Returns:
        An ``LLMResponse`` with text and/or tool calls.
    """
    if os.getenv("AGENT_LLM_MOCK"):
        return _mock_response(messages, tools)

    client = _get_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_completion_tokens,
    }
    if tools:
        kwargs["tools"] = tools

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0].message
    calls: list[ToolCall] = []
    for tc in choice.tool_calls or []:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

    return LLMResponse(content=choice.content or "", tool_calls=calls, raw=resp)
