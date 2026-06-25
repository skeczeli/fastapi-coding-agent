"""Tests for the Explorer subagent (#C2)."""

from agent.agents.explorer import _RecordingReadFile
from agent.state import TaskState
from agent.tools import get


def test_recording_read_file_records_source():
    """_RecordingReadFile records a Source(origin='repo') on successful read."""
    state = TaskState(request="explore")
    inner = get("read_file")
    wrapper = _RecordingReadFile(inner=inner, state=state)

    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world")
        path = f.name

    try:
        result = wrapper.execute({"path": path})
        assert result == "hello world"
        assert len(state.sources) == 1
        assert state.sources[0].origin == "repo"
        assert state.sources[0].ref == path
        assert state.sources[0].snippet == "hello world"
    finally:
        os.unlink(path)


def test_recording_read_file_skips_on_error():
    """_RecordingReadFile does NOT record a source when read_file fails."""
    state = TaskState(request="explore")
    inner = get("read_file")
    wrapper = _RecordingReadFile(inner=inner, state=state)

    result = wrapper.execute({"path": "/nonexistent/file.txt"})
    assert result.startswith("[error]")
    assert len(state.sources) == 0


from agent.agents import Subagent
from agent import llm
from agent.llm import LLMResponse, ToolCall


def test_explorer_satisfies_subagent_protocol():
    from agent.agents.explorer import Explorer
    assert isinstance(Explorer(), Subagent)


def test_explorer_run_calls_harness_and_stores_result(monkeypatch):
    """Explorer drives the LLM via harness.run_loop and stores the result."""
    from agent.agents.explorer import Explorer
    from agent.tools import base as base_mod

    state = TaskState(request="add a POST /users endpoint")

    monkeypatch.setattr(base_mod, "_read_file", lambda args: "# My Project")
    monkeypatch.setattr(base_mod, "_list_files", lambda args: "README.md\nsrc/\ntests/")

    llm.set_mock_script([
        LLMResponse(tool_calls=[
            ToolCall(id="1", name="list_files", arguments={"path": "."}),
        ]),
        LLMResponse(tool_calls=[
            ToolCall(id="2", name="read_file", arguments={"path": "README.md"}),
        ]),
        LLMResponse(content="## Project Overview\nA FastAPI app."),
    ])

    result = Explorer().run(state, "add a POST /users endpoint")

    assert result == "## Project Overview\nA FastAPI app."
    assert state.subagent_results["explorer"] == result
    repo_sources = [s for s in state.sources if s.origin == "repo"]
    assert any("README.md" in s.ref for s in repo_sources)


def test_explorer_no_tools_returns_direct_answer():
    """When the LLM answers without calling tools, result is still stored."""
    from agent.agents.explorer import Explorer

    state = TaskState(request="explore")

    llm.set_mock_script([
        LLMResponse(content="## Project Overview\nSimple project."),
    ])

    result = Explorer().run(state, "explore")

    assert result == "## Project Overview\nSimple project."
    assert state.subagent_results["explorer"] == result
