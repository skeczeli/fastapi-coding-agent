"""Tests for observability / tracing (#B4).

All offline: a recording tracer captures the spans/logs the instrumentation emits
(no Langfuse account needed), and a fake Langfuse client checks the SDK mapping.
"""

from contextlib import contextmanager

import pytest

from agent import harness, llm, observability
from agent.llm import LLMResponse, ToolCall
from agent.observability import LangfuseTracer, NoopTracer
from agent.state import TaskState


class RecordingTracer:
    """A Tracer that records span names and log payloads instead of sending them."""

    def __init__(self):
        self.spans: list[str] = []
        self.logs: list[dict] = []

    @contextmanager
    def span(self, name: str, **attrs):
        self.spans.append(name)
        yield

    def log(self, **fields):
        self.logs.append(fields)


@pytest.fixture(autouse=True)
def restore_tracer():
    """Snapshot/restore the global tracer so a test can't leak into others."""
    original = observability.get_tracer()
    try:
        yield
    finally:
        observability.set_tracer(original)


@pytest.fixture
def recording_tracer():
    rec = RecordingTracer()
    observability.set_tracer(rec)
    return rec


# --- the contract / wiring -------------------------------------------------


def test_recording_tracer_satisfies_protocol():
    assert isinstance(RecordingTracer(), observability.Tracer)


def test_init_tracer_is_noop_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert isinstance(observability.init_tracer(), NoopTracer)


# --- harness instrumentation (offline, via mock LLM) -----------------------


def test_harness_emits_turn_and_tool_spans(recording_tracer):
    state = TaskState(request="x")

    class _Spy:
        name = "spy"
        description = "d"
        parameters = {"type": "object", "properties": {}}
        permission = "read"

        def execute(self, args):
            return "ok"

    llm.set_mock_script(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="spy", arguments={})]),
            LLMResponse(content="done"),
        ]
    )
    harness.run_loop("sys", [_Spy()], state, "go")

    assert "agent.turn" in recording_tracer.spans
    assert "tool:spy" in recording_tracer.spans


def test_harness_flags_blocked_tool_as_warning(recording_tracer):
    state = TaskState(request="x")
    # The LLM asks for a tool not in the loop's set -> harness refuses -> WARNING.
    llm.set_mock_script(
        [
            LLMResponse(tool_calls=[ToolCall(id="1", name="ghost", arguments={})]),
            LLMResponse(content="stopped"),
        ]
    )
    harness.run_loop("sys", [], state, "go")

    assert any(log.get("level") == "WARNING" for log in recording_tracer.logs)


# --- llm instrumentation (offline, via a fake OpenAI client) ---------------


def test_llm_complete_emits_generation_with_usage(monkeypatch, recording_tracer):
    import types

    monkeypatch.delenv("AGENT_LLM_MOCK", raising=False)  # take the real (traced) path
    fake_resp = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="hi", tool_calls=[]))],
        usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=2, total_tokens=7),
    )
    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kw: fake_resp)
        )
    )
    monkeypatch.setattr(llm, "_get_client", lambda: fake_client)

    resp = llm.complete([{"role": "user", "content": "ping"}])

    assert resp.content == "hi"
    assert "llm.complete" in recording_tracer.spans
    usage_logs = [l for l in recording_tracer.logs if l.get("usage_details")]
    assert usage_logs and usage_logs[0]["usage_details"]["total"] == 7


# --- LangfuseTracer SDK mapping (fake client, no network) ------------------


def test_langfuse_tracer_maps_span_and_log():
    calls: list[tuple[str, dict]] = []

    class _FakeCM:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class _FakeClient:
        def start_as_current_observation(self, **kwargs):
            calls.append(("observation", kwargs))
            return _FakeCM()

        def update_current_span(self, **kwargs):
            calls.append(("update_span", kwargs))

        def update_current_generation(self, **kwargs):
            calls.append(("update_generation", kwargs))

    tracer = LangfuseTracer(_FakeClient())
    with tracer.span("tool:x", as_type="tool", input={"a": 1}):
        tracer.log(output="done")  # -> span
        tracer.log(output="hi", usage_details={"total": 3})  # -> generation

    kinds = [c[0] for c in calls]
    assert kinds[0] == "observation"
    assert "update_span" in kinds
    assert "update_generation" in kinds


def test_langfuse_tracer_never_raises_on_backend_error():
    class _BoomClient:
        def start_as_current_observation(self, **kwargs):
            raise RuntimeError("backend down")

        def update_current_span(self, **kwargs):
            raise RuntimeError("backend down")

        def update_current_generation(self, **kwargs):
            raise RuntimeError("backend down")

    tracer = LangfuseTracer(_BoomClient())
    # Must degrade silently — observability can't crash the agent.
    with tracer.span("x"):
        tracer.log(output="y")
