"""Implementer subagent — stub (#0). Real impl: #C3 (Dev 2).

Applies code changes based on Explorer + Researcher findings; has write access;
records the changed files in state.
"""

from __future__ import annotations

from agent.state import TaskState


class Implementer:
    name = "implementer"
    allowed_tools = ["read_file", "write_file", "list_files"]

    def run(self, state: TaskState, task: str) -> str:
        result = f"[implementer stub] would write code for: {task}"
        state.subagent_results[self.name] = result
        return result
