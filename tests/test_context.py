"""Context management tests (#C7)."""

from agent.context import detect_loop


class TestDetectLoop:
    def test_no_loop_returns_none(self):
        obs = ["read_file: line1", "write_file: ok", "run_command: passed"]
        assert detect_loop(obs) is None

    def test_empty_observations_returns_none(self):
        assert detect_loop([]) is None

    def test_identical_consecutive_detected(self):
        obs = [
            "read_file: contents of main.py",
            "read_file: contents of main.py",
            "read_file: contents of main.py",
        ]
        result = detect_loop(obs, window=3)
        assert result is not None
        assert "read_file" in result

    def test_loop_with_different_tools_not_detected(self):
        obs = [
            "read_file: file1 contents",
            "write_file: ok",
            "read_file: file2 contents",
            "write_file: ok",
        ]
        assert detect_loop(obs, window=4) is None

    def test_same_tool_different_results_not_detected(self):
        obs = [
            "read_file: file_a contents",
            "read_file: file_b contents",
            "read_file: file_c contents",
        ]
        assert detect_loop(obs, window=3) is None

    def test_loop_detected_only_in_window(self):
        obs = [
            "list_files: dir1",
            "list_files: dir2",
            "read_file: same",
            "read_file: same",
            "read_file: same",
        ]
        result = detect_loop(obs, window=4)
        assert result is not None

    def test_suggestion_text_is_actionable(self):
        obs = ["run_command: error: not found"] * 3
        result = detect_loop(obs, window=3)
        assert result is not None
        assert "try" in result.lower() or "different" in result.lower() or "strategy" in result.lower()

    def test_pair_repetition_detected(self):
        """Two-step cycle: A, B, A, B counts as a loop."""
        obs = [
            "read_file: main.py content",
            "run_command: error",
            "read_file: main.py content",
            "run_command: error",
        ]
        result = detect_loop(obs, window=4)
        assert result is not None
