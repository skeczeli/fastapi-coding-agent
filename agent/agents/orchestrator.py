"""Orchestrator / main agent — stub (#0). Real impl: #C1 (Dev 1, mine).

Owns the shared ``TaskState`` and coordinates the subagents via the
agents-as-tools pattern (explore → research → implement → test → review; the LLM
decides the real order). #0 ships only the entry-point signature so other lanes
can integrate against it; the loop is built in #C1.
"""

from __future__ import annotations

from agent.state import TaskState


def run(request: str) -> TaskState:
    """Run a full coding task end-to-end and return the final shared state.

    Stub for #0 — builds an empty ``TaskState`` and returns it. The real
    orchestration loop is implemented in #C1.
    """
    state = TaskState(request=request)
    state.note("[orchestrator stub] not yet wired — implemented in #C1")
    return state
