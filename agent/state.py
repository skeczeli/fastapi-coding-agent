"""Shared task state — the single object the orchestrator owns and every subagent reads/writes.

Defined in #0 as the contract all lanes code against. Kept as plain dataclasses
(stdlib) because this state is internal to a run, not validated external input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Origin labels for source attribution — the assignment requires every piece of
# info to be tagged by where it came from (RAG first, web second, etc.).
Origin = Literal["rag", "web", "repo", "memory", "inference"]


@dataclass
class Source:
    """A single piece of evidence with its provenance.

    Attributes:
        origin: Where the info came from (see ``Origin``).
        ref: Pointer to the source — a file path, URL, or doc section.
        snippet: Optional excerpt of the relevant text.
        score: Optional relevance score (e.g. from RAG retrieval).
    """

    origin: Origin
    ref: str
    snippet: str = ""
    score: float | None = None


@dataclass
class TaskState:
    """Shared state for one coding task, owned by the orchestrator.

    Attributes:
        request: The original user request, verbatim.
        plan: Ordered steps the agent intends to take.
        progress: Human-readable log of what has been done so far.
        subagent_results: Last result returned by each subagent, keyed by name.
        sources: All sources consulted, with origin labels (for attribution).
        files_modified: Paths the agent created or edited.
        observations: Free-form notes (errors, decisions, dead-ends).
    """

    request: str
    plan: list[str] = field(default_factory=list)
    progress: list[str] = field(default_factory=list)
    subagent_results: dict[str, Any] = field(default_factory=dict)
    sources: list[Source] = field(default_factory=list)
    files_modified: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)

    def add_source(
        self, origin: Origin, ref: str, snippet: str = "", score: float | None = None
    ) -> Source:
        """Record a consulted source and return it. Keeps attribution centralized."""
        src = Source(origin=origin, ref=ref, snippet=snippet, score=score)
        self.sources.append(src)
        return src

    def note(self, observation: str) -> None:
        """Append a free-form observation to the run log."""
        self.observations.append(observation)
