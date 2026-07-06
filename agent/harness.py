"""The shared harness loop — the one primitive the orchestrator and every subagent reuse.

Two nested loops (the in-class TP model, ported here in #A1):

- **Inner loop** (``run_loop`` / ``_drive``): one agent turn. Ask the LLM, and
  while it requests tools, validate each call against policy, execute it, feed the
  result back, and repeat until the LLM stops asking or ``max_iters`` is hit (the
  cap prevents infinite loops — a TP1 lesson).
- **Outer loop** (``converse``): the conversation across user turns. It keeps a
  single growing message history so the user can follow up, correct, or switch
  tasks without losing what the agent already did. ``python -m agent`` wires this
  to the real LLM + the base tools.

Both share ``_drive`` so the tool-dispatch logic lives in one place. ``run_loop``'s
signature, the by-object dispatch, and the policy/context hook points are #0
contracts — unchanged here.
"""

from __future__ import annotations

import json
from typing import Callable

from agent import config as config_mod, context, llm, observability, policy, tools
from agent.modes import HarnessMode, check_supervision, run_plan_approval
from agent.state import TaskState

# Default role prompt for the single-agent REPL (the ported in-class harness).
# The orchestrator has its own prompt; this one drives a lone agent with the five
# base tools, as in TP1.
SINGLE_AGENT_PROMPT = """You are a coding agent that resolves tasks by using tools.
You can read and write files, run shell commands, list directories, and search the
web. Work step by step: inspect what you need, make the change, and verify it (run
tests or commands) before reporting back. When the task is done, reply with a short
summary and stop calling tools."""


_loaded_config: config_mod.Config | None = None


def _get_config() -> config_mod.Config:
    """Lazily load and cache the agent config."""
    global _loaded_config
    if _loaded_config is None:
        _loaded_config = config_mod.load_config()
    return _loaded_config


def _drive(
    messages: list[dict],
    by_name: dict[str, tools.Tool],
    schemas: list[dict] | None,
    state: TaskState,
    max_iters: int,
    approval_fn: Callable[[str], bool] | None = None,
    mode: HarnessMode | None = None,
    supervision_fn: Callable[[str], bool] | None = None,
) -> str:
    """Inner loop: run tools over ``messages`` until the LLM gives a final answer.

    Mutates ``messages`` in place (so the outer ``converse`` loop keeps the full
    turn in its persistent history) and returns the LLM's final text — or a
    ``"[harness] stopped: ..."`` sentinel when it hits the iteration cap or a
    no-progress loop.

    Dispatch is by *object*: only tools in ``by_name`` are reachable, which is what
    enforces an agent's allowed-tools subset and lets transient tools (the
    orchestrator's subagent-as-tool adapters) work without global registration.
    """
    # Per-call outcome digests for *this* turn — the loop-detection signal
    # ("same call → same result"). Kept local on purpose: ``state.observations``
    # is shared across every subagent, so feeding it whole to detect_loop would
    # mix runs and invite cross-agent false positives.
    observations: list[str] = []
    tracer = observability.get_tracer()

    # One root span per agent turn; LLM generations and tool spans nest under it.
    # The last message is this turn's user request — surface it as the trace input.
    user_input = messages[-1].get("content") if messages else None
    with tracer.span("agent.turn", as_type="agent", input=user_input):
        for iteration in range(max_iters):
            # Keep the working context small (no-op until #C7). Assign back in place
            # so a summarized history propagates to the caller's persistent list.
            messages[:] = context.summarize_history(messages)
            resp = llm.complete(messages, tools=schemas)

            if not resp.tool_calls:
                tracer.log(output=resp.content, metadata={"iterations": iteration + 1})
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
                # SEAM(#A4): plan/supervision modes hook in here — confirm a write/
                # command (or show a plan) before the call runs. Read-only tools pass.
                with tracer.span(f"tool:{call.name}", as_type="tool", input=call.arguments):
                    tool = by_name.get(call.name)
                    if tool is None:
                        # Outside this loop's allowed toolset — refuse instead of
                        # falling through to the global registry.
                        result = f"[harness] unknown tool: {call.name}"
                    else:
                        denied = policy.check(tool, call.arguments, _get_config(), approval_fn)
                        if denied is not None:
                            result = f"[policy] denied: {denied}"
                        else:
                            if mode and supervision_fn:
                                sup_denied = check_supervision(
                                    tool, call.arguments, mode, supervision_fn
                                )
                                if sup_denied is not None:
                                    result = f"[supervision] denied: {sup_denied}"
                                else:
                                    result = tool.execute(call.arguments)
                            else:
                                result = tool.execute(call.arguments)
                    blocked = result.startswith(
                        ("[harness] unknown", "[policy] denied", "[supervision] denied")
                    )
                    tracer.log(output=result, level="WARNING" if blocked else None)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                observations.append(f"{call.name}: {result[:200]}")

            # Bail out of a no-progress loop instead of burning iterations (no-op
            # until #C7). This and the summarize call above are the context-management
            # hook points #C7 fills in once it lands.
            loop_suggestion = context.detect_loop(observations)
            if loop_suggestion is not None:
                state.note(f"run_loop stopped: {loop_suggestion}")
                tracer.log(output="[loop detected]", level="WARNING")
                return f"[harness] stopped: {loop_suggestion}"

        state.note(f"run_loop hit max_iters={max_iters}")
        tracer.log(
            output="[max iters]", level="WARNING", metadata={"iterations": max_iters}
        )
        return "[harness] stopped: reached max iterations"


def run_loop(
    system_prompt: str,
    tool_list: list[tools.Tool],
    state: TaskState,
    user_msg: str,
    max_iters: int = 25,
    approval_fn: Callable[[str], bool] | None = None,
    mode: HarnessMode | None = None,
    supervision_fn: Callable[[str], bool] | None = None,
) -> str:
    """Run one agent turn until the LLM finishes or hits the cap.

    The inner loop primitive reused by the orchestrator and every subagent. Builds
    a fresh message history from ``system_prompt`` + ``user_msg`` and drives it to
    a final answer. For a persistent multi-turn chat, use ``converse``.

    Args:
        system_prompt: Role/instructions for this agent.
        tool_list: Tools this agent is allowed to call (dispatch is restricted to these).
        state: Shared task state (read/written by tools and the loop).
        user_msg: The task/message that kicks off the turn.
        max_iters: Hard cap on tool-call iterations.
        approval_fn: Optional callback for require_approval checks.
        mode: Optional HarnessMode for plan/supervision modes.
        supervision_fn: Optional callback for supervision checks.

    Returns:
        The LLM's final text response (or a ``"[harness] stopped: ..."`` sentinel).
    """
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    by_name = {t.name: t for t in tool_list}
    schemas = tools.schemas(tool_list) if tool_list else None
    return _drive(messages, by_name, schemas, state, max_iters, approval_fn, mode, supervision_fn)


def converse(
    tool_list: list[tools.Tool],
    state: TaskState,
    system_prompt: str = SINGLE_AGENT_PROMPT,
    max_iters: int = 25,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    approval_fn: Callable[[str], bool] | None = None,
    mode: HarnessMode | None = None,
    supervision_fn: Callable[[str], bool] | None = None,
) -> None:
    """Outer conversation loop — the interactive REPL core.

    Seeds one system prompt, then for each user message runs the inner loop over
    the *same* growing ``messages`` list, so context (and everything the agent did)
    persists between turns. Returns when the user sends EOF or an exit command.

    ``input_fn``/``output_fn`` are injected (default stdin/stdout) so the loop can
    be driven in tests without a real terminal.
    """
    if approval_fn is None:
        def approval_fn(description: str) -> bool:
            output_fn(f"[approval required] {description}")
            try:
                answer = input_fn("Approve? [y/n]: ")
                return answer.strip().lower() in ("y", "yes", "s", "si")
            except (EOFError, KeyboardInterrupt):
                return False

    if mode is None:
        mode = HarnessMode()

    # Default supervision_fn — always defined so it's armed when /supervision is
    # toggled ON mid-session; check_supervision no-ops while the mode is OFF.
    if supervision_fn is None and mode:
        def supervision_fn(description: str) -> bool:
            output_fn(f"[supervision] {description}")
            try:
                answer = input_fn("Allow? [y/n]: ")
                return answer.strip().lower() in ("y", "yes", "s", "si")
            except (EOFError, KeyboardInterrupt):
                return False

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    by_name = {t.name: t for t in tool_list}
    schemas = tools.schemas(tool_list) if tool_list else None

    while True:
        try:
            user = input_fn("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("")
            return
        if user.lower() in {"exit", "quit", ":q"}:
            return
        if not user:
            continue

        # Toggle commands
        if user == "/plan":
            mode.plan_enabled = not mode.plan_enabled
            status = "ON" if mode.plan_enabled else "OFF"
            output_fn(f"Plan mode: {status}")
            continue
        if user == "/supervision":
            mode.supervision_enabled = not mode.supervision_enabled
            status = "ON" if mode.supervision_enabled else "OFF"
            output_fn(f"Supervision mode: {status}")
            continue

        messages.append({"role": "user", "content": user})

        # Plan mode gate: get approval before executing
        if mode:
            plan_result = run_plan_approval(
                user_msg=user,
                messages=messages,
                mode=mode,
                output_fn=output_fn,
                input_fn=input_fn,
            )
            if plan_result == "rejected":
                output_fn("[plan] Rejected — skipping execution.")
                messages.append({"role": "assistant", "content": "[plan rejected by user]"})
                continue
            if plan_result is not None:
                # User modified: replace the user message with new instructions
                messages[-1] = {"role": "user", "content": plan_result}

        # A failed turn (provider outage, rate limit that outlasted the retries)
        # shouldn't kill the whole session: report it, roll history back to
        # before this turn (drops the user message and any partial tool
        # exchanges — an assistant msg with unanswered tool_calls would poison
        # every later call), and let the user retry.
        turn_start = len(messages) - 1  # index of this turn's user message
        try:
            reply = _drive(
                messages, by_name, schemas, state, max_iters, approval_fn, mode, supervision_fn
            )
        except Exception as err:  # noqa: BLE001 — REPL boundary: report, keep chatting.
            del messages[turn_start:]
            output_fn(f"[turn failed] {type(err).__name__}: {err}")
            continue
        # Keep the assistant's final text in history so follow-ups have context.
        messages.append({"role": "assistant", "content": reply})
        output_fn(reply)
