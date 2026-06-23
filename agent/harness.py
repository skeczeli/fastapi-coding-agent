"""The shared harness loop — the one primitive the orchestrator and every subagent reuse.

⚠️  PROVISIONAL / NOT FINAL.  This is the *minimal* loop defined in #0 only to
    unblock #C1 (orchestrator) and let lanes code against a real signature. The
    production harness — history management between turns, supervision/plan
    modes, refactor of the TP1 notebook loop — is **#A1 (Dev 2)** and will
    replace the body below. Do not build heavy logic on top of it; build on the
    *signature* of ``run_loop``.

``run_loop`` drives one agent turn: ask the LLM, and while it requests tools,
validate each call against policy, execute it, feed the result back, and repeat
until the LLM stops asking or ``max_iters`` is hit (the cap prevents infinite
loops — a TP1 lesson).
"""

from __future__ import annotations

import json

from agent import llm, tools
from agent.state import TaskState


def _check_policy(tool: tools.Tool, args: dict) -> str | None:
    """Validate a tool call before execution. Returns an error string if denied.

    TODO(#A3): real policy engine (deny globs, workspace confinement,
    require-approval). Stub for #0 — always allows.
    """
    return None


def run_loop(
    system_prompt: str,
    tool_list: list[tools.Tool],
    state: TaskState,
    user_msg: str,
    max_iters: int = 25,
) -> str:
    """Run the agent loop until the LLM finishes its turn or hits the cap.

    PROVISIONAL primitive (see module docstring) — hardened in #A1.

    Args:
        system_prompt: Role/instructions for this agent.
        tool_list: Tools this agent is allowed to call.
        state: Shared task state (read/written by tools and the loop).
        user_msg: The task/message that kicks off the turn.
        max_iters: Hard cap on tool-call iterations.

    Returns:
        The LLM's final text response.
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    schemas = tools.schemas([t.name for t in tool_list]) if tool_list else None

    for _ in range(max_iters):
        resp = llm.complete(messages, tools=schemas)

        if not resp.tool_calls:
            return resp.content

        # Record the assistant turn that requested the tools.
        messages.append(
            {
                "role": "assistant",
                "content": resp.content or None,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in resp.tool_calls
                ],
            }
        )

        for call in resp.tool_calls:
            tool = tools.get(call.name)
            denied = _check_policy(tool, call.arguments)
            if denied is not None:
                result = f"[policy] denied: {denied}"
            else:
                result = tool.execute(call.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    state.note(f"run_loop hit max_iters={max_iters}")
    return "[harness] stopped: reached max iterations"
