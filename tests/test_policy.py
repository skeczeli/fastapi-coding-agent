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
