"""Plan mode + supervision mode — toggleable harness hooks (#A4).

Two independent modes that hook into the harness loop:
- Supervision: confirm before write/command tools; read-only pass freely.
- Plan mode: generate a step plan for user approval before executing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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
