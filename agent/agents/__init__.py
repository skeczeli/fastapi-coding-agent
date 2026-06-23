"""Subagent interface (#0).

A subagent is a specialized agent (Explorer, Researcher, Implementer, Tester,
Reviewer) the orchestrator calls via the *agents-as-tools* pattern. Each runs
its own harness loop with a focused prompt and a restricted set of tools, and
communicates results through the shared ``TaskState``.

The interface is a ``Protocol`` so each subagent can be a plain class without a
mandatory base. Real subagents live in #C2–#C5/#B3; #C1 owns the orchestrator.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent.state import TaskState


@runtime_checkable
class Subagent(Protocol):
    """A specialized agent the orchestrator coordinates.

    Attributes:
        name: Identifier used by the orchestrator (e.g. "explorer").
        allowed_tools: Tool names this subagent may call (its permission subset).
    """

    name: str
    allowed_tools: list[str]

    def run(self, state: TaskState, task: str) -> str:
        """Perform ``task``, mutate ``state`` with findings, return a summary."""
        ...
