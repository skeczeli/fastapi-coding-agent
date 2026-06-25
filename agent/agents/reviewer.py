"""Reviewer subagent (#C5, Dev 2).

Judges whether the implemented changes satisfy the *original* request and returns
an approve/reject verdict with a rationale. Runs its own harness loop with read +
command access (``read_file``, ``run_command`` — e.g. to inspect ``git diff``); it
never edits code or re-runs the suite (that's the Implementer's and Tester's jobs).

The agent is asked to end with a ``VERDICT: APPROVE/REJECT`` line; that verdict is
parsed back out and recorded in ``state.observations`` so downstream code (the
orchestrator's stop/replan decision) has a structured signal, not just prose.
"""

from __future__ import annotations

from agent import harness, tools
from agent.state import TaskState

REVIEWER_PROMPT = """You are the Reviewer in a multi-agent coding agent specialized \
in FastAPI. You decide whether the changes actually satisfy the ORIGINAL request. \
You do not edit code or run the test suite yourself — the Tester already did.

How you work:
- Re-read the original request and compare it to what changed: read the modified \
files and, when useful, inspect the diff (e.g. `git diff`).
- Weigh the Tester's report: failing checks are grounds to send the work back.
- Be specific. If you reject, state exactly what is missing or wrong and what must \
change to pass review.

End your reply with a single verdict line, exactly one of:
VERDICT: APPROVE - <one-line reason>
VERDICT: REJECT - <one-line reason>"""


def _verdict(text: str) -> str:
    """Extract the approve/reject verdict from the reviewer's final reply."""
    upper = text.upper()
    if "VERDICT: APPROVE" in upper:
        return "approve"
    if "VERDICT: REJECT" in upper:
        return "reject"
    return "unclear"  # model didn't follow the verdict format


def _context_preamble(state: TaskState) -> str:
    """Render the changes + prior reports so the reviewer can judge them."""
    parts: list[str] = []
    if state.files_modified:
        files = "\n".join(f"- {f}" for f in state.files_modified)
        parts.append(f"### files changed\n{files}")
    impl = state.subagent_results.get("implementer")
    if impl:
        parts.append(f"### implementer summary\n{impl}")
    report = state.subagent_results.get("tester")
    if report:
        parts.append(f"### tester report\n{report}")
    return "\n\n".join(parts)


class Reviewer:
    name = "reviewer"
    allowed_tools = ["read_file", "run_command"]

    def run(self, state: TaskState, task: str) -> str:
        preamble = _context_preamble(state)
        user_msg = f"Original request: {state.request}\n\nReview task: {task}"
        if preamble:
            user_msg += f"\n\n{preamble}"

        tool_list = [tools.get(name) for name in self.allowed_tools]
        result = harness.run_loop(
            system_prompt=REVIEWER_PROMPT,
            tool_list=tool_list,
            state=state,
            user_msg=user_msg,
        )
        state.subagent_results[self.name] = result
        state.note(f"reviewer verdict: {_verdict(result)}")
        return result
