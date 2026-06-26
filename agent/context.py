"""Context management — stubs (#0).

Keeps the working context small (summarize old history, retain key decisions)
and detects no-progress loops (same command → same error, re-reading files
without new info) so the agent can change strategy / stop / ask for help.
Real implementation is #C7 (Dev 3); #0 only fixes the interface.
"""

from __future__ import annotations


def summarize_history(messages: list[dict], keep_last: int = 6) -> list[dict]:
    """Compress old turns into a summary, keeping the last few verbatim.

    Stub for #0 — returns ``messages`` unchanged. Implemented in #C7.
    """
    return messages


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
