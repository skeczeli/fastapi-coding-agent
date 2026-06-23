"""Orchestrator tests (#C1) — offline, via the scripted LLM mock.

Proves the agents-as-tools wiring: the orchestrator exposes each subagent as a
tool, drives them through the shared ``run_loop`` against one ``TaskState``, and
that the loop dispatches over the toolset it's handed (so allowed-tools subsets
are actually enforced). All deterministic — no API key, no real subagents.
"""

from __future__ import annotations

from agent import llm
from agent.agents import orchestrator
from agent.agents.orchestrator import SubagentTool, default_subagents
from agent.state import TaskState


def _call(name: str, task: str, call_id: str = "c1") -> llm.LLMResponse:
    """An LLMResponse that asks to call one subagent-tool."""
    return llm.LLMResponse(
        content="",
        tool_calls=[llm.ToolCall(id=call_id, name=name, arguments={"task": task})],
    )


def _text(content: str) -> llm.LLMResponse:
    """A final LLMResponse with no tool calls (ends the loop)."""
    return llm.LLMResponse(content=content, tool_calls=[])


def test_subagent_tool_adapts_protocol_and_shares_state():
    state = TaskState(request="req")
    explorer = default_subagents()[0]
    tool = SubagentTool(subagent=explorer, state=state)

    assert tool.name == "explorer"
    assert tool.parameters["required"] == ["task"]
    out = tool.execute({"task": "map the repo"})

    # The adapter forwards to the subagent, which writes to the shared state.
    assert state.subagent_results["explorer"] == out
    assert any("explorer" in p for p in state.progress)


def test_orchestrator_runs_full_subagent_sequence():
    # Script the orchestrator LLM to walk explore→…→review, then summarize.
    llm.set_mock_script(
        [
            _call("explorer", "understand the repo", "c1"),
            _call("researcher", "how to validate email in FastAPI", "c2"),
            _call("implementer", "add POST /users", "c3"),
            _call("tester", "run pytest", "c4"),
            _call("reviewer", "check vs request", "c5"),
            _text("Done: added POST /users; sources: rag(docs/body.md)."),
        ]
    )

    state = orchestrator.run("add a POST /users endpoint")

    # Every subagent ran and recorded a result on the shared state.
    for name in ("explorer", "researcher", "implementer", "tester", "reviewer"):
        assert name in state.subagent_results
    assert "Done: added POST /users" in state.subagent_results["orchestrator"]
    # Progress log captured the delegation order.
    delegated = [p for p in state.progress if p.startswith("→")]
    assert [p.split()[1].rstrip(":") for p in delegated] == [
        "explorer",
        "researcher",
        "implementer",
        "tester",
        "reviewer",
    ]


def test_orchestrator_rejects_tool_outside_its_set():
    # The LLM asks for a base tool the orchestrator wasn't handed; the loop must
    # refuse rather than fall through to the global registry.
    llm.set_mock_script(
        [
            _call("write_file", "sneak a write", "c1"),
            _text("stopped"),
        ]
    )

    state = orchestrator.run("do something")

    # write_file is registered globally but not in the orchestrator's toolset,
    # so it was never executed (no files recorded) and the run still finished.
    assert state.files_modified == []
    assert state.subagent_results["orchestrator"] == "stopped"


def test_orchestrator_no_script_returns_state():
    # With no scripted tool calls the mock echoes and the loop returns at once —
    # keeps the #0 smoke-test contract (run returns a populated TaskState).
    state = orchestrator.run("trivial task")
    assert isinstance(state, TaskState)
    assert state.request == "trivial task"
    assert "orchestrator" in state.subagent_results
