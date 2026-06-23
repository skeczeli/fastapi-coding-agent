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

from agent import context, llm, tools
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
    # Dispatch over the tools this loop was handed, not the global registry.
    # This is what actually enforces a subagent's allowed-tools subset (the LLM
    # can't reach a tool outside ``tool_list``) and lets transient tools — like
    # the orchestrator's subagent-as-tool adapters — work without being
    # registered globally. NOTE(#A1, Dev 2): adopt this when hardening; the
    # signature is unchanged, only resolution moved from name→registry to object.
    by_name = {t.name: t for t in tool_list}
    schemas = tools.schemas(tool_list) if tool_list else None

    # Per-call outcome digests for *this* loop — the loop-detection signal
    # ("same call → same result"). Kept local on purpose: ``state.observations``
    # is shared across every subagent, so feeding it whole to detect_loop would
    # mix runs and invite cross-agent false positives.
    observations: list[str] = []

    for _ in range(max_iters):
        # Keep the working context small (no-op until #C7, Dev 3).
        messages = context.summarize_history(messages)
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
            tool = by_name.get(call.name)
            if tool is None:
                # Outside this loop's allowed toolset — refuse instead of
                # falling through to the global registry.
                result = f"[harness] unknown tool: {call.name}"
            else:
                denied = _check_policy(tool, call.arguments)
                if denied is not None:
                    result = f"[policy] denied: {denied}"
                else:
                    result = tool.execute(call.arguments)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            observations.append(f"{call.name}: {result[:200]}")

        # Bail out of a no-progress loop instead of burning iterations (no-op
        # until #C7, Dev 3). NOTE(#A1): this and the summarize call above are the
        # context-management hook points #C7 fills in once it lands.
        if context.detect_loop(observations):
            state.note("run_loop stopped: no-progress loop detected")
            return "[harness] stopped: no progress (loop detected)"

    state.note(f"run_loop hit max_iters={max_iters}")
    return "[harness] stopped: reached max iterations"
