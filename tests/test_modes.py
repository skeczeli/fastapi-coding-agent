"""Tests for plan mode + supervision mode (#A4)."""

from dataclasses import dataclass, field

from agent.modes import HarnessMode, check_supervision


@dataclass
class _FakeTool:
    name: str = "fake"
    description: str = ""
    parameters: dict = field(default_factory=dict)
    permission: str = "read"

    def execute(self, args: dict) -> str:
        return ""


def test_supervision_allows_read_tools_without_prompt():
    mode = HarnessMode(supervision_enabled=True)
    tool = _FakeTool(permission="read")
    result = check_supervision(tool, {}, mode, confirm_fn=lambda _: False)
    assert result is None


def test_supervision_blocks_write_when_rejected():
    mode = HarnessMode(supervision_enabled=True)
    tool = _FakeTool(name="write_file", permission="write")
    result = check_supervision(tool, {"path": "x.py"}, mode, confirm_fn=lambda _: False)
    assert result is not None
    assert "rejected" in result


def test_supervision_allows_write_when_approved():
    mode = HarnessMode(supervision_enabled=True)
    tool = _FakeTool(name="write_file", permission="write")
    result = check_supervision(tool, {"path": "x.py"}, mode, confirm_fn=lambda _: True)
    assert result is None


def test_supervision_blocks_command_when_rejected():
    mode = HarnessMode(supervision_enabled=True)
    tool = _FakeTool(name="run_command", permission="command")
    result = check_supervision(tool, {"command": "rm foo"}, mode, confirm_fn=lambda _: False)
    assert result is not None


def test_supervision_noop_when_disabled():
    mode = HarnessMode(supervision_enabled=False)
    tool = _FakeTool(name="write_file", permission="write")
    result = check_supervision(tool, {}, mode, confirm_fn=lambda _: False)
    assert result is None


# Plan mode tests

from agent.modes import run_plan_approval


def test_plan_approval_noop_when_disabled():
    mode = HarnessMode(plan_enabled=False)
    result = run_plan_approval(
        user_msg="add endpoint",
        messages=[],
        mode=mode,
        output_fn=lambda _: None,
        input_fn=lambda _: "",
    )
    assert result is None


def test_plan_approval_shows_plan_and_accepts():
    mode = HarnessMode(plan_enabled=True)
    outputs: list[str] = []
    # Simulate: LLM returns a plan text, user approves
    import agent.llm as llm_mod
    from agent.llm import LLMResponse

    llm_mod.set_mock_script([LLMResponse(content="1. Read main.py\n2. Add function")])

    result = run_plan_approval(
        user_msg="add endpoint",
        messages=[{"role": "system", "content": "you are an agent"}],
        mode=mode,
        output_fn=outputs.append,
        input_fn=lambda _: "a",
    )
    assert result is None  # None means "proceed with execution"
    assert any("1. Read main.py" in o for o in outputs)


def test_plan_approval_rejects():
    mode = HarnessMode(plan_enabled=True)
    import agent.llm as llm_mod
    from agent.llm import LLMResponse

    llm_mod.set_mock_script([LLMResponse(content="1. Do something")])

    result = run_plan_approval(
        user_msg="add endpoint",
        messages=[{"role": "system", "content": "you are an agent"}],
        mode=mode,
        output_fn=lambda _: None,
        input_fn=lambda _: "r",
    )
    assert result == "rejected"


def test_plan_approval_modify_returns_new_message():
    mode = HarnessMode(plan_enabled=True)
    import agent.llm as llm_mod
    from agent.llm import LLMResponse

    llm_mod.set_mock_script([LLMResponse(content="1. Do X")])

    inputs = iter(["m", "do X but skip step 2"])
    result = run_plan_approval(
        user_msg="add endpoint",
        messages=[{"role": "system", "content": "you are an agent"}],
        mode=mode,
        output_fn=lambda _: None,
        input_fn=lambda _: next(inputs),
    )
    assert result == "do X but skip step 2"
