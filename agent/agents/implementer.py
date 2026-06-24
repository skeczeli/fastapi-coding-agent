"""Implementer subagent (#C3, Dev 2).

Applies concrete code changes based on the Explorer's and Researcher's findings.
Runs its own harness loop with write access (``read_file``, ``write_file``,
``list_files`` — no ``run_command``; running checks is the Tester's job, #C4).

Every successful write is recorded in ``state.files_modified`` via a thin adapter
around ``write_file``, so the changed-files "diff" is tracked automatically rather
than parsed out of the transcript.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent import harness, tools
from agent.state import TaskState

IMPLEMENTER_PROMPT = """You are the Implementer in a multi-agent coding agent \
specialized in FastAPI. You apply concrete code changes to satisfy the task, \
grounded in the Explorer's and Researcher's findings.

How you work:
- Inspect before editing: read the relevant files and list directories so your \
change fits the existing code (style, imports, conventions).
- Make focused, minimal edits — only what the task needs. ``write_file`` replaces \
a file's whole contents, so when changing an existing file, read it first and \
write back the full updated version.
- You do NOT run commands or tests; the Tester verifies your work afterwards.
- If the task is ambiguous or the findings are insufficient to implement safely, \
stop and explain what's missing instead of guessing.

When done, reply with a short summary of what you changed and why."""


@dataclass
class _RecordingWriteFile:
    """Wraps ``write_file`` to log successful writes into ``state.files_modified``.

    Delegates every Tool attribute to the wrapped tool so the harness sees an
    ordinary ``write_file``; only ``execute`` is augmented to record the path.
    """

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
        # The base write_file returns "[ok] ..." only on a successful write.
        if result.startswith("[ok]"):
            path = args.get("path")
            if path and path not in self.state.files_modified:
                self.state.files_modified.append(path)
        return result


def _findings_preamble(state: TaskState) -> str:
    """Render prior subagent findings + consulted sources to ground the edit."""
    parts: list[str] = []
    for role in ("explorer", "researcher"):
        result = state.subagent_results.get(role)
        if result:
            parts.append(f"### {role} findings\n{result}")
    if state.sources:
        refs = "\n".join(f"- [{s.origin}] {s.ref}" for s in state.sources)
        parts.append(f"### sources consulted\n{refs}")
    return "\n\n".join(parts)


class Implementer:
    name = "implementer"
    allowed_tools = ["read_file", "write_file", "list_files"]

    def _tools(self, state: TaskState) -> list[tools.Tool]:
        """Resolve allowed tool names to objects, wrapping write_file to record writes."""
        resolved: list[tools.Tool] = []
        for tool_name in self.allowed_tools:
            tool = tools.get(tool_name)
            if tool_name == "write_file":
                tool = _RecordingWriteFile(inner=tool, state=state)
            resolved.append(tool)
        return resolved

    def run(self, state: TaskState, task: str) -> str:
        preamble = _findings_preamble(state)
        user_msg = f"{task}\n\n{preamble}" if preamble else task

        result = harness.run_loop(
            system_prompt=IMPLEMENTER_PROMPT,
            tool_list=self._tools(state),
            state=state,
            user_msg=user_msg,
        )
        state.subagent_results[self.name] = result
        return result
