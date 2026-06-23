"""Persistent per-project memory — stub (#0).

Knowledge that outlives a single run: detected architecture, important files,
conventions, decisions, session summaries. Persists to disk and loads on
startup. Real implementation is #C6 (Dev 3); #0 only fixes the interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProjectMemory:
    """Key-value project memory. Load/save are no-ops until #C6."""

    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str = ".agent_memory.json") -> "ProjectMemory":
        """Load memory from disk (no-op stub — returns empty until #C6)."""
        return cls()

    def save(self, path: str = ".agent_memory.json") -> None:
        """Persist memory to disk (no-op stub until #C6)."""
        pass
