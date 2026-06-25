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
