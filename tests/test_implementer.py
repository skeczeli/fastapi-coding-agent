"""Tests for the Implementer subagent (#C3).

Drive the subagent's harness loop offline with a scripted LLM: it must apply a
write, record the changed file in shared state, and report a summary.
"""

import pytest

from agent.agents.implementer import Implementer
from agent.agents import Subagent
from agent import harness, llm
from agent.config import Config
from agent.llm import LLMResponse, ToolCall
from agent.state import TaskState


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Point the harness policy at a temp workspace so writes are allowed there.

    Since #A3, the policy confines writes to ``config.workspace``; tests that
    apply a write must run inside it. Overrides the harness's cached config.
    """
    monkeypatch.setattr(harness, "_loaded_config", Config(workspace=str(tmp_path)))
    return tmp_path


def test_implementer_satisfies_subagent_protocol():
    assert isinstance(Implementer(), Subagent)


def test_implementer_applies_write_and_records_file(workspace):
    state = TaskState(request="add a health endpoint")
    target = workspace / "main.py"
    # Turn 1: model writes the file. Turn 2: model gives its final summary.
    llm.set_mock_script(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="1",
                        name="write_file",
                        arguments={"path": str(target), "content": "x = 1\n"},
                    )
                ]
            ),
            LLMResponse(content="Created main.py with a health endpoint."),
        ]
    )

    result = Implementer().run(state, "create main.py")

    assert target.read_text() == "x = 1\n"  # the edit was applied
    assert str(target) in state.files_modified  # recorded in shared state
    assert state.subagent_results["implementer"] == result
    assert "main.py" in result


def test_implementer_does_not_record_failed_write(workspace):
    state = TaskState(request="x")
    # Writing into a path whose parent is a file -> base write_file returns [error].
    blocker = workspace / "afile"
    blocker.write_text("")
    bad = blocker / "nested.py"
    llm.set_mock_script(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="1", name="write_file", arguments={"path": str(bad), "content": "y"})
                ]
            ),
            LLMResponse(content="could not write"),
        ]
    )

    Implementer().run(state, "try a bad write")

    assert state.files_modified == []  # failed write must not be recorded
