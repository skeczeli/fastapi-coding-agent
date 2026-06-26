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


class _StubSubagent:
    """A subagent that records a result without calling the LLM.

    Keeps the orchestrator tests isolated from real subagent implementations:
    the scripted mock LLM is a shared FIFO, so a *real* subagent (e.g. the
    Implementer, #C3) running its own loop would consume the orchestrator's
    scripted turns. Stubs honour the test's stated intent ("no real subagents").
    """

    def __init__(self, name: str):
        self.name = name
        self.allowed_tools: list[str] = []

    def run(self, state: TaskState, task: str) -> str:
        result = f"[{self.name}] handled: {task}"
        state.subagent_results[self.name] = result
        return result


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

    # Inject pure stubs so only the orchestrator's loop consumes the script
    # (real subagents would pop their own LLM turns off the shared FIFO).
    roster = [
        _StubSubagent(n)
        for n in ("explorer", "researcher", "implementer", "tester", "reviewer")
    ]
    state = orchestrator.run("add a POST /users endpoint", subagents=roster)

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


from agent.agents.orchestrator import RememberProjectTool
from agent.memory import ProjectMemory


def test_remember_project_tool_satisfies_protocol():
    from agent.tools import Tool

    mem = ProjectMemory(path="/tmp/unused")
    tool = RememberProjectTool(memory=mem)
    assert isinstance(tool, Tool)
    assert tool.name == "remember_project"
    assert tool.permission == "write"


def test_remember_project_tool_writes_to_memory(tmp_path):
    mem_dir = str(tmp_path / ".agent_memory")
    mem = ProjectMemory(path=mem_dir)
    tool = RememberProjectTool(memory=mem)

    result = tool.execute({"category": "dependencies", "content": "SQLAlchemy for ORM"})
    assert "remembered" in result.lower() or "saved" in result.lower()
    assert mem.get_category("dependencies") == ["SQLAlchemy for ORM"]


def test_remember_project_tool_handles_architecture(tmp_path):
    mem_dir = str(tmp_path / ".agent_memory")
    mem = ProjectMemory(path=mem_dir)
    tool = RememberProjectTool(memory=mem)

    tool.execute({"category": "architecture", "content": "FastAPI monolith"})
    assert mem.data["architecture"] == "FastAPI monolith"


def test_orchestrator_includes_remember_project_tool():
    llm.set_mock_script(
        [
            _call("remember_project", "remember deps", "c1"),
            _text("Done."),
        ]
    )

    roster = [_StubSubagent(n) for n in ("explorer", "researcher", "implementer", "tester", "reviewer")]
    state = orchestrator.run("task", subagents=roster)
    assert "unknown tool" not in state.subagent_results.get("orchestrator", "")


# --- CLI entrypoint (#I1) ---------------------------------------------------


def test_render_state_shows_sources_with_origin_labels():
    state = TaskState(request="add GET /health")
    state.subagent_results["orchestrator"] = "Done."
    state.add_source("rag", "tutorial/first-steps.md", score=0.91)
    state.add_source("web", "https://fastapi.tiangolo.com/")
    state.files_modified.append("app/main.py")
    state.subagent_results["explorer"] = "FastAPI app with one router"

    report = orchestrator.render_state(state)

    # Origin labels are surfaced (the assignment's "differentiate origin" rule).
    assert "[rag] tutorial/first-steps.md" in report
    assert "score=0.91" in report
    assert "[web] https://fastapi.tiangolo.com/" in report
    assert "app/main.py" in report
    assert "explorer: FastAPI app with one router" in report
    assert "Done." in report


def test_render_state_handles_empty_state():
    report = orchestrator.render_state(TaskState(request="noop"))
    assert "Sources consulted: (none)" in report
    assert "Files modified: (none)" in report


def test_main_runs_end_to_end_in_mock(monkeypatch, capsys):
    monkeypatch.setenv("AGENT_LLM_MOCK", "1")
    llm.set_mock_script([_text("All done: added GET /health.")])

    rc = orchestrator.main(["add", "a", "GET", "/health", "--max-iters", "3"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "ORCHESTRATOR RESULT" in out
    assert "All done: added GET /health." in out


def test_main_reports_halt_with_exit_code_2(monkeypatch, capsys):
    monkeypatch.setenv("AGENT_LLM_MOCK", "1")
    # A subagent-tool call that loops the orchestrator into the stop sentinel.
    state_summary = "[harness] stopped: reached max iterations"
    monkeypatch.setattr(orchestrator, "run", lambda *a, **k: _halted_state(state_summary))

    rc = orchestrator.main(["do", "something"])
    assert rc == 2
    assert "ORCHESTRATOR RESULT" in capsys.readouterr().out


def _halted_state(summary: str) -> TaskState:
    state = TaskState(request="do something")
    state.subagent_results["orchestrator"] = summary
    return state
