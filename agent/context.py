"""Context management — stubs (#0).

Keeps the working context small (summarize old history, retain key decisions)
and detects no-progress loops (same command → same error, re-reading files
without new info) so the agent can change strategy / stop / ask for help.
Real implementation is #C7 (Dev 3); #0 only fixes the interface.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

TOKEN_BUDGET = 12_000
WORD_TOKEN_FACTOR = 1.3


def _estimate_tokens(messages: list[dict]) -> int:
    """Cheap token estimate: count words × 1.3."""
    total = 0
    for m in messages:
        content = m.get("content") or ""
        total += len(content.split())
    return int(total * WORD_TOKEN_FACTOR)


def _build_mechanical_summary(messages: list[dict]) -> str:
    """Extract key decisions and tool results from discarded messages."""
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or ""
        if role == "assistant" and content:
            parts.append(f"- Assistant: {content[:200]}")
        elif role == "tool":
            parts.append(f"- Tool result: {content[:150]}")
    if not parts:
        return "Prior conversation context (details truncated)."
    return "Summary of earlier conversation:\n" + "\n".join(parts[:15])


def _llm_summary(messages: list[dict]) -> str:
    """Ask the LLM to summarize discarded messages."""
    from agent import llm as llm_mod

    text_parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content") or ""
        if content:
            text_parts.append(f"{role}: {content[:300]}")
    combined = "\n".join(text_parts[:20])
    summary_messages = [
        {"role": "system", "content": "Summarize this conversation fragment in 2-3 sentences. Keep key decisions, findings, and tool results. Be concise."},
        {"role": "user", "content": combined},
    ]
    try:
        resp = llm_mod.complete(summary_messages, tools=None)
        return resp.content or _build_mechanical_summary(messages)
    except Exception:
        log.warning("LLM summary failed, falling back to mechanical summary")
        return _build_mechanical_summary(messages)


def summarize_history(
    messages: list[dict], keep_last: int = 6, token_budget: int = TOKEN_BUDGET
) -> list[dict]:
    """Compress old turns when approaching the token budget.

    Below budget: returns messages unchanged.
    Over budget: keeps the system message + a summary of old turns + the last
    *keep_last* messages. Uses mechanical truncation for moderate overages;
    calls the LLM for a real summary when the overflow is large (>2× budget).
    """
    if _estimate_tokens(messages) <= token_budget:
        return messages

    system = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    if len(non_system) <= keep_last:
        return messages

    to_discard = non_system[:-keep_last]
    to_keep = non_system[-keep_last:]

    # Never split an assistant(tool_calls) → tool exchange: a kept tool message
    # whose tool_calls request was discarded is rejected by the API (OpenAI 400:
    # "role 'tool' must be a response to a preceding message with 'tool_calls'").
    while to_keep and to_keep[0].get("role") == "tool":
        to_discard.append(to_keep.pop(0))

    overflow = _estimate_tokens(messages) - token_budget
    if overflow > token_budget:
        summary_text = _llm_summary(to_discard)
    else:
        summary_text = _build_mechanical_summary(to_discard)

    summary_msg = {"role": "user", "content": f"[context summary]\n{summary_text}"}
    return system + [summary_msg] + to_keep


def detect_loop(observations: list[str], window: int = 4) -> str | None:
    """Detect no-progress loops in recent observations.

    Checks the last *window* observations for two patterns:
    1. Same observation repeated ≥3 times (single-step loop).
    2. A repeating cycle of 2 observations (two-step loop: A,B,A,B).

    Returns a suggestion string when a loop is found, None otherwise.
    """
    if len(observations) < 3:
        return None

    recent = observations[-window:]

    # Pattern 1: single observation repeated ≥3 times in the window
    from collections import Counter
    counts = Counter(recent)
    for obs, count in counts.most_common(1):
        if count >= 3:
            tool_name = obs.split(":")[0] if ":" in obs else "unknown"
            return (
                f"No progress: {tool_name} returned the same result {count} times. "
                f"Try a different approach — change the arguments, use a different tool, "
                f"or ask the user for help."
            )

    # Pattern 2: two-step cycle (A, B, A, B) — need ≥4 observations
    if len(recent) >= 4:
        pairs = [(recent[i], recent[i + 1]) for i in range(len(recent) - 1)]
        if len(pairs) >= 3:
            for i in range(len(pairs) - 2):
                if pairs[i] == pairs[i + 2]:
                    tool_a = pairs[i][0].split(":")[0] if ":" in pairs[i][0] else "unknown"
                    tool_b = pairs[i][1].split(":")[0] if ":" in pairs[i][1] else "unknown"
                    return (
                        f"No progress: repeating cycle of {tool_a} → {tool_b} "
                        f"with the same results. Try a different strategy or ask for help."
                    )

    return None
