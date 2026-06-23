"""Tester subagent — stub (#0). Real impl: #C4 (Dev 2).

Runs group-defined checks (pytest / start server & hit endpoint / lint / build),
reports pass/fail + output to state.
"""

from __future__ import annotations

from agent.state import TaskState


class Tester:
    name = "tester"
    allowed_tools = ["run_command", "read_file"]

    def run(self, state: TaskState, task: str) -> str:
        result = f"[tester stub] would run checks for: {task}"
        state.subagent_results[self.name] = result
        return result
