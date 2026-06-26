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

# Output-token cap. gpt-5-nano needs a high cap because it spends internal
# reasoning tokens — too low and it returns empty content (TP1 lesson). The model
# id and the cap's *parameter name* now come from ``providers.chat_config()`` so
# the backend (OpenAI / Gemini / any OpenAI-compatible endpoint) is env-selectable.
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

# Optional FIFO queue of canned responses for mock mode. Lets tests drive a
# multi-step agent loop (e.g. the orchestrator calling subagents in sequence)
# deterministically without an API key. ``None`` → fall back to the echo mock.
_mock_script: list["LLMResponse"] | None = None


def set_mock_script(responses: list["LLMResponse"] | None) -> None:
    """Queue canned ``LLMResponse``s consumed (FIFO) by mock mode.

    Each ``complete()`` call in mock mode pops the next scripted response; once
    the queue is empty it reverts to the default echo behaviour. Pass ``None``
    to clear the script. Tests must clear it when done (an autouse fixture does
    this) so a script can't leak into other tests — same global-state hazard as
    the tool registry.
    """
    global _mock_script
    _mock_script = list(responses) if responses is not None else None


def _get_client():
    """Lazily build the (OpenAI-compatible) client for the configured chat provider.

    Importing this module needs no API key. The client is pointed at whatever
    ``providers.chat_config()`` resolves (OpenAI by default; Gemini or any
    compatible endpoint when the ``AGENT_LLM_*`` env vars are set).
    """
    global _client
    if _client is None:
        from dotenv import load_dotenv  # load .env here, not at import time

        # Pull keys/URLs from .env so every entrypoint (orchestrator, CLI, #I1)
        # works without the caller remembering to load it. No-op if already in the
        # environment; never reached in mock mode.
        load_dotenv()
        from openai import OpenAI  # imported lazily; core import stays light

        from agent.providers import chat_config

        _client = OpenAI(**chat_config().client_kwargs())
    return _client


def _mock_response(messages: list[dict], tools: list[dict] | None) -> LLMResponse:
    """Deterministic stand-in used when AGENT_LLM_MOCK is set.

    Pops the next scripted response if one was queued via ``set_mock_script``;
    otherwise echoes the last message so simple smoke tests still work.
    """
    if _mock_script:
        return _mock_script.pop(0)
    last = messages[-1]["content"] if messages else ""
    return LLMResponse(content=f"[mock] received: {last}", tool_calls=[], raw=None)


def complete(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> LLMResponse:
    """Call the LLM once and return a normalized ``LLMResponse``.

    Args:
        messages: OpenAI-style chat messages.
        tools: Tool schemas in OpenAI tool-calling format (see ``tools.schemas``).
        model: Model id. ``None`` → the configured provider's default
            (``providers.chat_config().model``).
        max_completion_tokens: Output cap; kept high for reasoning models. Sent
            under the provider's expected param name (``max_completion_tokens`` for
            OpenAI, ``max_tokens`` for most compatible endpoints).

    Returns:
        An ``LLMResponse`` with text and/or tool calls.
    """
    if os.getenv("AGENT_LLM_MOCK"):
        return _mock_response(messages, tools)

    import time

    from agent import observability
    from agent.providers import chat_config

    cfg = chat_config()
    model = model or cfg.model

    tracer = observability.get_tracer()
    client = _get_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        cfg.max_tokens_param: max_completion_tokens,
    }
    if tools:
        kwargs["tools"] = tools

    # One generation span per LLM call: model + prompt in, response + token usage
    # + latency out. Nests under whatever span the harness opened (the agent turn).
    start = time.perf_counter()
    with tracer.span("llm.complete", as_type="generation", model=model, input=messages):
        resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0].message
        calls: list[ToolCall] = []
        for tc in choice.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        usage = getattr(resp, "usage", None)
        usage_details = None
        if usage is not None:
            usage_details = {
                k: v
                for k, v in {
                    "input": getattr(usage, "prompt_tokens", None),
                    "output": getattr(usage, "completion_tokens", None),
                    "total": getattr(usage, "total_tokens", None),
                }.items()
                if v is not None
            }
        tracer.log(
            output=choice.content or "",
            usage_details=usage_details,
            metadata={
                "latency_ms": round((time.perf_counter() - start) * 1000),
                "tool_calls": len(calls),
            },
        )

    return LLMResponse(content=choice.content or "", tool_calls=calls, raw=resp)
