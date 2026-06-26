"""Tests for the harness conversation loop + REPL core (#A1).

Drive ``converse`` with injected input/output and a scripted/faked LLM so the two
nested loops run offline: the outer loop must keep history across user turns, and
the inner loop must run a tool end-to-end.
"""

from dataclasses import dataclass, field

from agent import harness, llm
from agent.llm import LLMResponse, ToolCall
from agent.state import TaskState


@dataclass
class _SpyTool:
    """A registerable Tool that records the args it was called with."""

    calls: list = field(default_factory=list)
    name: str = "spy"
    description: str = "records its calls"
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    permission: str = "read"

    def execute(self, args: dict) -> str:
        self.calls.append(args)
        return "tool-ran"


def _feed(*turns: str):
    """Build an input_fn that returns the given user turns, then 'exit'."""
    script = iter([*turns, "exit"])
    return lambda _prompt="": next(script)


def test_converse_runs_a_tool_end_to_end():
    state = TaskState(request="x")
    spy = _SpyTool()
    # Turn 1: the LLM asks for the tool, then (after the result) gives a final answer.
    llm.set_mock_script(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="spy", arguments={"k": "v"})]),
            LLMResponse(content="done"),
        ]
    )
    outputs: list[str] = []
    harness.converse([spy], state, input_fn=_feed("do the thing"), output_fn=outputs.append)

    assert spy.calls == [{"k": "v"}]  # the tool actually executed
    assert outputs == ["done"]  # the final answer reached the user


def test_converse_keeps_history_between_turns(monkeypatch):
    seen: list[list[dict]] = []

    def fake_complete(messages, tools=None, **kwargs):
        seen.append([dict(m) for m in messages])  # snapshot what the LLM saw
        return LLMResponse(content="ok")

    monkeypatch.setattr(harness.llm, "complete", fake_complete)
    state = TaskState(request="x")
    harness.converse([], state, input_fn=_feed("first", "second"), output_fn=lambda *_: None)

    # The second turn's context must still contain the first turn's user message
    # and be longer than the first — i.e. history persisted across user turns.
    assert any(m.get("content") == "first" for m in seen[1])
    assert len(seen[1]) > len(seen[0])


def test_converse_exits_on_command():
    state = TaskState(request="x")
    outputs: list[str] = []
    # Only 'exit' — the loop should return without ever calling the LLM.
    harness.converse([], state, input_fn=lambda _p="": "exit", output_fn=outputs.append)
    assert outputs == []


def test_run_loop_still_returns_final_text():
    # The refactor must not change run_loop's contract (mock echoes, no tool calls).
    state = TaskState(request="noop")
    out = harness.run_loop("you are a test agent", [], state, "do nothing")
    assert isinstance(out, str)
    assert "do nothing" in out


# Supervision mode tests

from agent.modes import HarnessMode
from agent import llm


def test_drive_supervision_blocks_write_tool():
    state = TaskState(request="x")
    spy = _SpyTool(name="writer", permission="write")
    mode = HarnessMode(supervision_enabled=True)

    llm.set_mock_script(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="writer", arguments={})]),
            LLMResponse(content="ok"),
        ]
    )
    outputs: list[str] = []
    harness.converse(
        [spy],
        state,
        mode=mode,
        input_fn=_feed("do it"),
        output_fn=outputs.append,
        approval_fn=lambda _: True,  # policy approval
        supervision_fn=lambda _: False,  # supervision rejects
    )
    assert spy.calls == []  # tool never executed


def test_drive_supervision_allows_read_tool():
    state = TaskState(request="x")
    spy = _SpyTool(name="reader", permission="read")
    mode = HarnessMode(supervision_enabled=True)

    llm.set_mock_script(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="reader", arguments={})]),
            LLMResponse(content="done"),
        ]
    )
    outputs: list[str] = []
    harness.converse(
        [spy],
        state,
        mode=mode,
        input_fn=_feed("read it"),
        output_fn=outputs.append,
        supervision_fn=lambda _: False,  # would reject, but reads bypass
    )
    assert spy.calls == [{}]  # tool executed despite supervision_fn returning False


# Plan mode tests


def test_converse_plan_mode_approve_then_execute():
    state = TaskState(request="x")
    spy = _SpyTool()
    mode = HarnessMode(plan_enabled=True)

    llm.set_mock_script(
        [
            # Plan LLM call returns a plan
            LLMResponse(content="1. Call spy tool"),
            # Execution: LLM calls the tool, then gives final answer
            LLMResponse(tool_calls=[ToolCall(id="1", name="spy", arguments={})]),
            LLMResponse(content="done"),
        ]
    )

    # input sequence: user msg, plan approval ("a"), then exit
    inputs = iter(["do task", "a", "exit"])
    outputs: list[str] = []

    harness.converse(
        [spy],
        state,
        mode=mode,
        input_fn=lambda _p="": next(inputs),
        output_fn=outputs.append,
    )
    assert spy.calls == [{}]  # tool executed after plan approved
    assert any("Plan" in o for o in outputs)


def test_converse_plan_mode_reject_skips_execution():
    state = TaskState(request="x")
    spy = _SpyTool()
    mode = HarnessMode(plan_enabled=True)

    llm.set_mock_script([LLMResponse(content="1. Call spy tool")])

    inputs = iter(["do task", "r", "exit"])
    outputs: list[str] = []

    harness.converse(
        [spy],
        state,
        mode=mode,
        input_fn=lambda _p="": next(inputs),
        output_fn=outputs.append,
    )
    assert spy.calls == []  # tool never ran


def test_converse_toggle_commands():
    state = TaskState(request="x")
    mode = HarnessMode()
    outputs: list[str] = []

    harness.converse(
        [],
        state,
        mode=mode,
        input_fn=_feed("/plan", "/supervision"),
        output_fn=outputs.append,
    )
    assert mode.plan_enabled is True
    assert mode.supervision_enabled is True
    assert any("plan mode" in o.lower() and "on" in o.lower() for o in outputs)
    assert any("supervision mode" in o.lower() and "on" in o.lower() for o in outputs)


# Loop detection tests


def test_loop_detection_stops_and_includes_suggestion(monkeypatch):
    """When detect_loop fires, the harness stops and the suggestion appears in output."""
    from agent import context

    monkeypatch.setattr(
        context, "detect_loop", lambda obs, **kw: "Try a different file."
    )

    state = TaskState(request="x")
    spy = _SpyTool()
    llm.set_mock_script(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="spy", arguments={})]),
            # Would continue, but loop detection should stop it before iteration 2.
        ]
    )
    result = harness.run_loop("agent", [spy], state, "go")
    assert "stopped" in result
    assert "Try a different file" in result
