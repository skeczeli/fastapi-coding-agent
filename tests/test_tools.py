"""Unit tests for the five base tools (#A2).

One test per tool plus error-path coverage. File ops use ``tmp_path`` so nothing
touches the real workspace; ``web_search`` is exercised only on its key-missing
path to stay offline and deterministic.
"""

from agent import tools
from agent.tools import base


def test_read_file(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("hola mundo")
    assert base.read_file.execute({"path": str(f)}) == "hola mundo"


def test_read_file_missing_returns_error(tmp_path):
    out = base.read_file.execute({"path": str(tmp_path / "nope.txt")})
    assert out.startswith("[error]")


def test_write_file_creates_parent_dirs(tmp_path):
    f = tmp_path / "sub" / "out.txt"
    out = base.write_file.execute({"path": str(f), "content": "data"})
    assert out.startswith("[ok]")
    assert f.read_text() == "data"


def test_run_command_captures_stdout():
    out = base.run_command.execute({"command": "echo hello"})
    assert "hello" in out
    assert "[exit 0]" in out


def test_run_command_reports_nonzero_exit():
    out = base.run_command.execute({"command": "exit 3"})
    assert "[exit 3]" in out


def test_list_files_marks_dirs(tmp_path):
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "subdir").mkdir()
    out = base.list_files.execute({"path": str(tmp_path)})
    assert "a.txt" in out
    assert "subdir/" in out


def test_web_search_without_key_returns_error(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    out = base.web_search.execute({"query": "fastapi dependency injection"})
    assert out.startswith("[error]")


def test_all_base_tools_registered_and_executable():
    for name in ("read_file", "write_file", "run_command", "list_files", "web_search"):
        tool = tools.get(name)
        assert callable(tool.execute)
