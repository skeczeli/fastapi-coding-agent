"""Tests for the policy engine (#A3)."""

from dataclasses import dataclass, field

from agent.config import Config, PolicyConfig


@dataclass
class _StubTool:
    """Minimal Tool stub for policy checks."""

    name: str = "stub"
    description: str = ""
    parameters: dict = field(default_factory=dict)
    permission: str = "read"
    calls: list = field(default_factory=list)

    def execute(self, args: dict) -> str:
        self.calls.append(args)
        return ""


def _config(**kwargs) -> Config:
    """Build a Config with overrides; everything else permissive."""
    return Config(
        workspace=kwargs.get("workspace", "."),
        read=kwargs.get("read", PolicyConfig()),
        write=kwargs.get("write", PolicyConfig()),
        commands=kwargs.get("commands", PolicyConfig()),
    )


# -- Glob matching --


class TestGlobToRegex:
    def test_exact_match(self):
        from agent.policy import _glob_to_regex

        assert _glob_to_regex(".env").match(".env")
        assert not _glob_to_regex(".env").match(".envx")

    def test_star_matches_within_segment(self):
        from agent.policy import _glob_to_regex

        assert _glob_to_regex(".env.*").match(".env.local")
        assert _glob_to_regex(".env.*").match(".env.production")
        assert not _glob_to_regex(".env.*").match(".env")

    def test_double_star_matches_across_segments(self):
        from agent.policy import _glob_to_regex

        assert _glob_to_regex("**/secrets/**").match("foo/secrets/key.txt")
        assert _glob_to_regex("**/*.lock").match("deep/nested/package.lock")
        assert _glob_to_regex(".git/**").match(".git/config")
        assert _glob_to_regex(".git/**").match(".git/refs/heads/main")


# -- Read deny --


class TestReadDeny:
    def test_denies_path_matching_read_deny(self):
        from agent.policy import check

        cfg = _config(read=PolicyConfig(deny=[".env", ".env.*"]))
        tool = _StubTool(permission="read")
        result = check(tool, {"path": ".env"}, cfg)
        assert result is not None
        assert ".env" in result

    def test_denies_dotenv_variant(self):
        from agent.policy import check

        cfg = _config(read=PolicyConfig(deny=[".env.*"]))
        tool = _StubTool(permission="read")
        result = check(tool, {"path": ".env.local"}, cfg)
        assert result is not None

    def test_denies_nested_secrets(self):
        from agent.policy import check

        cfg = _config(read=PolicyConfig(deny=["**/secrets/**"]))
        tool = _StubTool(permission="read")
        result = check(tool, {"path": "config/secrets/api.key"}, cfg)
        assert result is not None

    def test_allows_normal_path(self):
        from agent.policy import check

        cfg = _config(read=PolicyConfig(deny=[".env", ".env.*"]))
        tool = _StubTool(permission="read")
        result = check(tool, {"path": "src/main.py"}, cfg)
        assert result is None

    def test_allows_when_no_deny_patterns(self):
        from agent.policy import check

        cfg = _config()
        tool = _StubTool(permission="read")
        result = check(tool, {"path": ".env"}, cfg)
        assert result is None

    def test_no_path_arg_skips_check(self):
        from agent.policy import check

        cfg = _config(read=PolicyConfig(deny=[".env"]))
        tool = _StubTool(permission="read")
        result = check(tool, {"query": "fastapi"}, cfg)
        assert result is None


class TestWriteDeny:
    def test_denies_path_matching_write_deny(self):
        from agent.policy import check

        cfg = _config(write=PolicyConfig(deny=[".git/**"]))
        tool = _StubTool(permission="write")
        result = check(tool, {"path": ".git/config", "content": "x"}, cfg)
        assert result is not None
        assert ".git/**" in result

    def test_denies_lock_files(self):
        from agent.policy import check

        cfg = _config(write=PolicyConfig(deny=["**/*.lock"]))
        tool = _StubTool(permission="write")
        result = check(tool, {"path": "deep/nested/package.lock", "content": ""}, cfg)
        assert result is not None

    def test_allows_normal_write_inside_workspace(self, tmp_path):
        from agent.policy import check

        ws = tmp_path / "workspace"
        ws.mkdir()
        cfg = _config(workspace=str(ws))
        tool = _StubTool(permission="write")
        result = check(tool, {"path": str(ws / "main.py"), "content": "x"}, cfg)
        assert result is None


class TestWorkspaceConfinement:
    def test_denies_write_outside_workspace(self, tmp_path):
        from agent.policy import check

        ws = tmp_path / "workspace"
        ws.mkdir()
        cfg = _config(workspace=str(ws))
        tool = _StubTool(permission="write")
        result = check(tool, {"path": "/etc/passwd", "content": "x"}, cfg)
        assert result is not None
        assert "outside workspace" in result

    def test_allows_write_inside_workspace(self, tmp_path):
        from agent.policy import check

        ws = tmp_path / "workspace"
        ws.mkdir()
        cfg = _config(workspace=str(ws))
        tool = _StubTool(permission="write")
        result = check(tool, {"path": str(ws / "file.py"), "content": "x"}, cfg)
        assert result is None

    def test_blocks_path_traversal(self, tmp_path):
        from agent.policy import check

        ws = tmp_path / "workspace"
        ws.mkdir()
        cfg = _config(workspace=str(ws))
        tool = _StubTool(permission="write")
        sneaky = str(ws / ".." / "secrets" / "key.txt")
        result = check(tool, {"path": sneaky, "content": "x"}, cfg)
        assert result is not None
        assert "outside workspace" in result

    def test_read_not_confined_to_workspace(self, tmp_path):
        from agent.policy import check

        ws = tmp_path / "workspace"
        ws.mkdir()
        cfg = _config(workspace=str(ws))
        tool = _StubTool(permission="read")
        result = check(tool, {"path": "/some/other/file.py"}, cfg)
        assert result is None


class TestCommandDeny:
    def test_denies_rm_rf(self):
        from agent.policy import check

        cfg = _config(commands=PolicyConfig(deny=["rm -rf*"]))
        tool = _StubTool(permission="command")
        result = check(tool, {"command": "rm -rf /"}, cfg)
        assert result is not None
        assert "rm -rf" in result

    def test_denies_sudo(self):
        from agent.policy import check

        cfg = _config(commands=PolicyConfig(deny=["sudo*"]))
        tool = _StubTool(permission="command")
        result = check(tool, {"command": "sudo apt install foo"}, cfg)
        assert result is not None

    def test_denies_chained_with_and(self):
        from agent.policy import check

        cfg = _config(commands=PolicyConfig(deny=["rm -rf*"]))
        tool = _StubTool(permission="command")
        result = check(tool, {"command": "cd /tmp && rm -rf /"}, cfg)
        assert result is not None

    def test_denies_chained_with_semicolon(self):
        from agent.policy import check

        cfg = _config(commands=PolicyConfig(deny=["rm -rf*"]))
        tool = _StubTool(permission="command")
        result = check(tool, {"command": "echo hi; rm -rf /"}, cfg)
        assert result is not None

    def test_denies_chained_with_pipe(self):
        from agent.policy import check

        cfg = _config(commands=PolicyConfig(deny=["sudo*"]))
        tool = _StubTool(permission="command")
        result = check(tool, {"command": "cat file | sudo tee /etc/x"}, cfg)
        assert result is not None

    def test_allows_safe_command(self):
        from agent.policy import check

        cfg = _config(commands=PolicyConfig(deny=["rm -rf*", "sudo*"]))
        tool = _StubTool(permission="command")
        result = check(tool, {"command": "echo hello"}, cfg)
        assert result is None

    def test_denies_fork_bomb(self):
        from agent.policy import check

        cfg = _config(commands=PolicyConfig(deny=[":(){*"]))
        tool = _StubTool(permission="command")
        result = check(tool, {"command": ":(){ :|:& };:"}, cfg)
        assert result is not None


class TestRequireApproval:
    def test_calls_approval_fn_and_allows_on_true(self):
        from agent.policy import check

        cfg = _config(commands=PolicyConfig(require_approval=["git push*"]))
        tool = _StubTool(permission="command")
        called_with = []

        def approve(desc: str) -> bool:
            called_with.append(desc)
            return True

        result = check(tool, {"command": "git push origin main"}, cfg, approval_fn=approve)
        assert result is None
        assert len(called_with) == 1
        assert "git push origin main" in called_with[0]

    def test_denies_on_approval_fn_false(self):
        from agent.policy import check

        cfg = _config(commands=PolicyConfig(require_approval=["pip install*"]))
        tool = _StubTool(permission="command")
        result = check(
            tool, {"command": "pip install requests"}, cfg, approval_fn=lambda _: False
        )
        assert result is not None
        assert "rejected" in result

    def test_denies_when_no_approval_fn(self):
        from agent.policy import check

        cfg = _config(commands=PolicyConfig(require_approval=["git push*"]))
        tool = _StubTool(permission="command")
        result = check(tool, {"command": "git push origin main"}, cfg, approval_fn=None)
        assert result is not None
        assert "no approval handler" in result

    def test_uses_process_wide_default_when_no_explicit_fn(self):
        # The CLI installs a default handler; check() falls back to it when the
        # caller (e.g. a subagent's run_loop) passes no approval_fn.
        from agent.policy import check, set_approval_fn

        cfg = _config(commands=PolicyConfig(require_approval=["pip install*"]))
        tool = _StubTool(permission="command")
        try:
            set_approval_fn(lambda _: True)
            assert check(tool, {"command": "pip install requests"}, cfg) is None
            set_approval_fn(lambda _: False)
            assert "rejected" in check(tool, {"command": "pip install requests"}, cfg)
        finally:
            set_approval_fn(None)

    def test_explicit_fn_overrides_default(self):
        from agent.policy import check, set_approval_fn

        cfg = _config(commands=PolicyConfig(require_approval=["pip install*"]))
        tool = _StubTool(permission="command")
        try:
            set_approval_fn(lambda _: False)  # default would reject…
            # …but the explicit handler wins.
            assert check(tool, {"command": "pip install x"}, cfg, approval_fn=lambda _: True) is None
        finally:
            set_approval_fn(None)

    def test_deny_checked_before_approval(self):
        from agent.policy import check

        cfg = _config(
            commands=PolicyConfig(deny=["rm -rf*"], require_approval=["rm *"])
        )
        tool = _StubTool(permission="command")
        result = check(tool, {"command": "rm -rf /"}, cfg, approval_fn=lambda _: True)
        assert result is not None
        assert "deny" in result

    def test_approval_not_triggered_for_non_matching(self):
        from agent.policy import check

        cfg = _config(commands=PolicyConfig(require_approval=["git push*"]))
        tool = _StubTool(permission="command")
        called = []

        result = check(
            tool,
            {"command": "echo hello"},
            cfg,
            approval_fn=lambda d: called.append(d) or True,
        )
        assert result is None
        assert called == []


class TestHarnessIntegration:
    def test_harness_blocks_denied_tool_call(self, monkeypatch):
        from agent import harness, llm
        from agent.config import Config, PolicyConfig
        from agent.llm import LLMResponse, ToolCall
        from agent.state import TaskState

        cfg = Config(
            workspace=".",
            read=PolicyConfig(deny=[".env"]),
        )
        monkeypatch.setattr(harness, "_loaded_config", cfg)

        spy = _StubTool(name="read_file", permission="read")

        llm.set_mock_script(
            [
                LLMResponse(
                    tool_calls=[
                        ToolCall(id="1", name="read_file", arguments={"path": ".env"})
                    ]
                ),
                LLMResponse(content="I see it was denied"),
            ]
        )

        state = TaskState(request="read .env")
        harness.run_loop("test", [spy], state, "read .env")
        assert spy.calls == []  # tool was NOT executed
