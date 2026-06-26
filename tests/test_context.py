"""Context management tests (#C7)."""

from agent.context import detect_loop, summarize_history, _estimate_tokens


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


class TestEstimateTokens:
    def test_empty_messages(self):
        assert _estimate_tokens([]) == 0

    def test_counts_words_with_factor(self):
        msgs = [{"role": "user", "content": "hello world"}]
        tokens = _estimate_tokens(msgs)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_handles_none_content(self):
        msgs = [{"role": "assistant", "content": None}]
        assert _estimate_tokens(msgs) == 0


class TestSummarizeHistory:
    def test_short_history_unchanged(self):
        msgs = [
            {"role": "system", "content": "you are an agent"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = summarize_history(msgs)
        assert result == msgs

    def test_preserves_system_message(self):
        msgs = [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": f"msg {i}"} for i in range(20)
        ]
        result = summarize_history(msgs, keep_last=4)
        assert result[0] == {"role": "system", "content": "sys"}

    def test_keeps_last_n_messages(self):
        msgs = [{"role": "system", "content": "sys"}] + [
            {"role": "user", "content": f"msg {i}"} for i in range(20)
        ]
        result = summarize_history(msgs, keep_last=4)
        # Last 4 non-system messages should be preserved verbatim
        assert result[-4:] == msgs[-4:]

    def test_under_budget_returns_unchanged(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "short"},
        ]
        result = summarize_history(msgs, keep_last=6)
        assert result == msgs

    def test_over_budget_truncates(self):
        long_content = "word " * 5000  # ~6500 tokens
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": long_content},
            {"role": "user", "content": long_content},
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": "recent"},
        ]
        result = summarize_history(msgs, keep_last=2)
        # Should be shorter than original
        assert len(result) < len(msgs)
        # Last messages preserved
        assert result[-1]["content"] == "recent"

    def test_tool_messages_in_kept_window(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "1", "content": "result"},
            {"role": "assistant", "content": "done"},
        ]
        result = summarize_history(msgs, keep_last=4)
        assert result == msgs
