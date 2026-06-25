"""Plan mode + supervision mode — toggleable harness hooks (#A4).

Two independent modes that hook into the harness loop:
- Supervision: confirm before write/command tools; read-only pass freely.
- Plan mode: generate a step plan for user approval before executing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent import llm as llm_mod
from agent.tools import Tool


@dataclass
class HarnessMode:
    """Toggleable harness modes — both independent, both default off."""

    plan_enabled: bool = False
    supervision_enabled: bool = False


def check_supervision(
    tool: Tool,
    args: dict,
    mode: HarnessMode,
    confirm_fn: Callable[[str], bool],
) -> str | None:
    """Supervision hook: prompt user before write/command tools.

    Returns None to proceed, or a reason string to block the call.
    Read-only tools always pass. No-op when supervision is disabled.
    """
    if not mode.supervision_enabled:
        return None
    if tool.permission == "read":
        return None
    desc = f"[supervision] {tool.name}({args})"
    if confirm_fn(desc):
        return None
    return f"tool call rejected by user (supervision): {tool.name}"


_PLAN_PROMPT = (
    "Before executing, generate a step-by-step plan for the following task. "
    "List numbered steps. Do NOT execute anything yet — only output the plan."
)


def run_plan_approval(
    user_msg: str,
    messages: list[dict],
    mode: HarnessMode,
    output_fn: Callable[[str], None],
    input_fn: Callable[[str], str],
) -> str | None:
    """Plan mode hook: ask the LLM for a plan, let the user approve/modify/reject.

    Returns:
        None  — plan approved, proceed with execution.
        "rejected" — user rejected, abort this turn.
        str — modified instructions from the user (re-plan with these).
    """
    if not mode.plan_enabled:
        return None

    plan_messages = [m for m in messages if m.get("role") == "system"]
    plan_messages.append({"role": "user", "content": f"{_PLAN_PROMPT}\n\nTask: {user_msg}"})

    resp = llm_mod.complete(plan_messages)
    plan_text = resp.content

    output_fn(f"\n--- Proposed Plan ---\n{plan_text}\n---------------------")
    output_fn("[a]pprove / [m]odify / [r]eject")

    choice = input_fn("plan> ").strip().lower()
    if choice.startswith("a"):
        return None
    if choice.startswith("r"):
        return "rejected"
    # modify: ask for new instructions
    new_instructions = input_fn("new instructions> ").strip()
    return new_instructions
