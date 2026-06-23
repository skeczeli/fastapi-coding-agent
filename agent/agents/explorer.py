"""Explorer subagent — stub (#0). Real impl: #C2 (Dev 3).

Understands the target repo (structure, deps, conventions, relevant files),
read-only tools, writes findings to shared state.
"""

from __future__ import annotations

from agent.state import TaskState


class Explorer:
    name = "explorer"
    allowed_tools = ["read_file", "list_files"]

    def run(self, state: TaskState, task: str) -> str:
        result = f"[explorer stub] would explore the repo for: {task}"
        state.subagent_results[self.name] = result
        return result
