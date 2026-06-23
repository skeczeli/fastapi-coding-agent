"""Reviewer subagent — stub (#0). Real impl: #C5 (Dev 2).

Reviews the diff/changes against the original request; approves or sends back
with reasons.
"""

from __future__ import annotations

from agent.state import TaskState


class Reviewer:
    name = "reviewer"
    allowed_tools = ["read_file", "run_command"]

    def run(self, state: TaskState, task: str) -> str:
        result = f"[reviewer stub] would review changes against: {task}"
        state.subagent_results[self.name] = result
        return result
