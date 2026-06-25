"""Explorer subagent (#C2).

Understands the target repo: structure, architecture, dependencies,
conventions, relevant files. Read-only tools only (``read_file``,
``list_files``). Writes a structured summary to shared state.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent import tools
from agent.state import TaskState


@dataclass
class _RecordingReadFile:
    """Wraps ``read_file`` to record each file read as a repo source."""

    inner: tools.Tool
    state: TaskState

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def description(self) -> str:
        return self.inner.description

    @property
    def parameters(self) -> dict:
        return self.inner.parameters

    @property
    def permission(self) -> str:
        return self.inner.permission

    def execute(self, args: dict) -> str:
        result = self.inner.execute(args)
        path = args.get("path", "")
        if not result.startswith("[error]"):
            self.state.add_source("repo", path, snippet=result[:200])
        return result


class Explorer:
    name = "explorer"
    allowed_tools = ["read_file", "list_files"]

    def run(self, state: TaskState, task: str) -> str:
        result = f"[explorer stub] would explore the repo for: {task}"
        state.subagent_results[self.name] = result
        return result
