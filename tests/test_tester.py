"""Tests for the Tester subagent (#C4).

Drive the subagent offline with a scripted LLM and real shell commands ('true' /
'false') so the pass/fail recording is exercised without touching the network.
"""

from agent.agents.tester import Tester
from agent.agents import Subagent
from agent import llm
from agent.llm import LLMResponse, ToolCall
from agent.state import TaskState


def test_tester_satisfies_subagent_protocol():
    assert isinstance(Tester(), Subagent)


def test_tester_records_passing_check():
    state = TaskState(request="verify the endpoint")
    llm.set_mock_script(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="1", name="run_command", arguments={"command": "true"})]
            ),
            LLMResponse(content="All checks passed."),
        ]
    )

    result = Tester().run(state, "run the test suite")

    assert result == "All checks passed."
    assert state.subagent_results["tester"] == result
    assert any("pass" in note for note in state.observations)


def test_tester_records_failing_check():
    state = TaskState(request="verify the endpoint")
    llm.set_mock_script(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="1", name="run_command", arguments={"command": "false"})]
            ),
            LLMResponse(content="A check failed."),
        ]
    )

    Tester().run(state, "run the test suite")

    assert any("fail" in note for note in state.observations)


def test_tester_does_not_write_files(tmp_path):
    # The tester's toolset has no write_file, so a write request is refused by the
    # harness (unknown tool) and nothing is created.
    state = TaskState(request="x")
    target = tmp_path / "sneaky.py"
    llm.set_mock_script(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="1", name="write_file", arguments={"path": str(target), "content": "x"})
                ]
            ),
            LLMResponse(content="cannot write"),
        ]
    )

    Tester().run(state, "try to write")

    assert not target.exists()
    assert state.files_modified == []
