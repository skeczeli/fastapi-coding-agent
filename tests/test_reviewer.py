"""Tests for the Reviewer subagent (#C5).

Drive the subagent offline with a scripted LLM: it must surface an approve/reject
verdict, record it on shared state, and be able to inspect changes via a command.
"""

from agent.agents.reviewer import Reviewer
from agent.agents import Subagent
from agent import llm
from agent.llm import LLMResponse, ToolCall
from agent.state import TaskState


def test_reviewer_satisfies_subagent_protocol():
    assert isinstance(Reviewer(), Subagent)


def test_reviewer_approves_and_records_verdict():
    state = TaskState(request="add a POST /users endpoint")
    state.files_modified.append("main.py")
    state.subagent_results["tester"] = "All checks passed."
    llm.set_mock_script(
        [LLMResponse(content="Matches the request.\nVERDICT: APPROVE - meets the request")]
    )

    result = Reviewer().run(state, "review the changes against the request")

    assert "APPROVE" in result
    assert state.subagent_results["reviewer"] == result
    assert any("reviewer verdict: approve" in note for note in state.observations)


def test_reviewer_rejects_and_records_verdict():
    state = TaskState(request="add a POST /users endpoint")
    state.subagent_results["tester"] = "1 failed: missing email validation."
    llm.set_mock_script(
        [LLMResponse(content="Email is not validated.\nVERDICT: REJECT - tests fail")]
    )

    Reviewer().run(state, "review the changes")

    assert any("reviewer verdict: reject" in note for note in state.observations)


def test_reviewer_can_inspect_changes_via_command():
    state = TaskState(request="x")
    # Turn 1: the reviewer inspects the diff. Turn 2: it gives its verdict.
    llm.set_mock_script(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="1", name="run_command", arguments={"command": "echo somediff"})
                ]
            ),
            LLMResponse(content="VERDICT: APPROVE - diff looks right"),
        ]
    )

    result = Reviewer().run(state, "review")

    assert "APPROVE" in result
