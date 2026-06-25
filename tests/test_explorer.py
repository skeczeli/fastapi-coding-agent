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
