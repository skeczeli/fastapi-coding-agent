"""Explorer subagent (#C2).

Understands the target repo: structure, architecture, dependencies,
conventions, relevant files. Read-only tools only (``read_file``,
``list_files``). Writes a structured summary to shared state.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent import harness, tools
from agent.state import TaskState


EXPLORER_PROMPT = """\
You are the Explorer in a multi-agent coding agent specialized in FastAPI.
Your job is to understand a target repository's structure, architecture,
dependencies, conventions, and identify files relevant to the task.

Strategy:
1. Start by calling ``list_files`` on the project root to see the top-level structure.
2. Read key files: README, pyproject.toml, setup.py, requirements.txt, config files.
3. Drill into source directories — list them, then read entry points and key modules.
4. Identify naming conventions, patterns, and framework usage.
5. Pay special attention to files relevant to the user's specific task.

Your answer MUST be a structured summary with these sections:
- **Project Overview**: what the project does, framework, language version.
- **Structure**: directory layout with purpose of key directories.
- **Architecture**: main components, entry points, data flow.
- **Dependencies**: key libraries and their roles.
- **Conventions**: naming patterns, project-specific idioms, test conventions.
- **Relevant Files**: files most relevant to the user's task, with a short note on why."""


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

    def _tools(self, state: TaskState) -> list[tools.Tool]:
        resolved: list[tools.Tool] = []
        for tool_name in self.allowed_tools:
            tool = tools.get(tool_name)
            if tool_name == "read_file":
                tool = _RecordingReadFile(inner=tool, state=state)
            resolved.append(tool)
        return resolved

    def run(self, state: TaskState, task: str) -> str:
        result = harness.run_loop(
            system_prompt=EXPLORER_PROMPT,
            tool_list=self._tools(state),
            state=state,
            user_msg=task,
        )
        state.subagent_results[self.name] = result
        return result
