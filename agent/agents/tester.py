"""Tester subagent (#C4, Dev 2).

Validates the implementer's work by running group-defined checks (pytest, lint,
build, or starting the server and hitting an endpoint) and reporting pass/fail.
Runs its own harness loop with read + command access (``run_command``,
``read_file``); it never writes — fixing failures is the Implementer's job.

Each command is run through a thin adapter that reads its exit code and records a
structured pass/fail note in ``state.observations``, so the run leaves an auditable
trail of which checks were run and how they turned out.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent import harness, tools
from agent.state import TaskState

TESTER_PROMPT = """You are the Tester in a multi-agent coding agent specialized in \
FastAPI. You verify that the implementer's changes actually work — you do not edit \
code.

How you work:
- Decide which checks fit the task: run the test suite (e.g. `pytest`), a linter, a \
build, or start the app and hit the relevant endpoint.
- Run checks with run_command and read files to interpret failures. Prefer the \
project's own commands/config over inventing new ones.
- Report clearly: which checks you ran, whether each passed or failed, and the key \
output (especially error messages) for any failure.
- Do NOT try to fix the code; hand failures back so the implementer can address them.

When done, reply with a concise pass/fail verdict and the evidence behind it."""


def _verdict(result: str) -> str:
    """Classify a run_command result by its exit-code prefix."""
    if result.startswith("[exit 0]"):
        return "pass"
    if result.startswith("[exit "):
        return "fail"
    return "error"  # timeout, empty command, etc.


@dataclass
class _RecordingRunCommand:
    """Wraps ``run_command`` to log each check's pass/fail into shared state.

    Delegates every Tool attribute to the wrapped tool so the harness sees an
    ordinary ``run_command``; only ``execute`` is augmented to record the outcome.
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
        command = args.get("command", "")
        self.state.note(f"tester check: {command!r} -> {_verdict(result)}")
        return result


def _context_preamble(state: TaskState) -> str:
    """Render what the implementer changed so the tester knows what to verify."""
    parts: list[str] = []
    if state.files_modified:
        files = "\n".join(f"- {f}" for f in state.files_modified)
        parts.append(f"### files changed by the implementer\n{files}")
    impl = state.subagent_results.get("implementer")
    if impl:
        parts.append(f"### implementer summary\n{impl}")
    return "\n\n".join(parts)


class Tester:
    name = "tester"
    allowed_tools = ["run_command", "read_file"]

    def _tools(self, state: TaskState) -> list[tools.Tool]:
        """Resolve allowed tool names to objects, wrapping run_command to record checks."""
        resolved: list[tools.Tool] = []
        for tool_name in self.allowed_tools:
            tool = tools.get(tool_name)
            if tool_name == "run_command":
                tool = _RecordingRunCommand(inner=tool, state=state)
            resolved.append(tool)
        return resolved

    def run(self, state: TaskState, task: str) -> str:
        preamble = _context_preamble(state)
        user_msg = f"{task}\n\n{preamble}" if preamble else task

        result = harness.run_loop(
            system_prompt=TESTER_PROMPT,
            tool_list=self._tools(state),
            state=state,
            user_msg=user_msg,
        )
        state.subagent_results[self.name] = result
        return result
