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


def detect_loop(observations: list[str], window: int = 3) -> bool:
    """Return True if the recent observations look like a no-progress loop.

    Stub for #0 — always False. Implemented in #C7.
    """
    return False
