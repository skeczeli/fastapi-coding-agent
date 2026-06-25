"""Observability — tracer interface, a no-op default, and a Langfuse tracer (#B4).

The contract (#0) lets other code wrap LLM calls, tools, and retrievals with
spans/logs through two methods — ``span`` and ``log`` — without knowing which
backend is active. ``NoopTracer`` is the default; ``LangfuseTracer`` (#B4) sends
a full trace (prompts, model, tokens, latency, cost, tool calls, errors) to
Langfuse when ``LANGFUSE_*`` keys are configured.

Tracing must never break the agent: every Langfuse call is guarded, so a backend
error degrades to "no trace", not a crash.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager, contextmanager
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class Tracer(Protocol):
    """Minimal tracing surface other modules instrument against."""

    def span(self, name: str, **attrs) -> AbstractContextManager[None]:
        """Context manager wrapping a unit of work (LLM call, tool, retrieval)."""
        ...

    def log(self, **fields) -> None:
        """Record structured data on the current span (output, tokens, latency...)."""
        ...


class NoopTracer:
    """Does nothing — the default until a real tracer is installed."""

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[None]:
        yield

    def log(self, **fields) -> None:
        pass

    def flush(self) -> None:
        pass


# Attributes ``start_as_current_observation`` understands; anything else a caller
# passes to ``span`` is folded into ``metadata`` instead of dropped.
_OBSERVATION_KWARGS = frozenset(
    {
        "as_type",
        "input",
        "output",
        "metadata",
        "model",
        "model_parameters",
        "usage_details",
        "cost_details",
        "level",
        "status_message",
        "version",
    }
)


class LangfuseTracer:
    """Tracer backed by the Langfuse SDK (#B4).

    Maps the project's two-method interface onto Langfuse v4: ``span`` opens a
    ``start_as_current_observation`` context (so nested spans/generations attach
    automatically), and ``log`` updates the current observation. Every call is
    wrapped in try/except — observability must not crash the agent.
    """

    def __init__(self, client) -> None:
        self._client = client

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[None]:
        cm = None
        try:
            kwargs = {k: v for k, v in attrs.items() if k in _OBSERVATION_KWARGS}
            extra = {k: v for k, v in attrs.items() if k not in _OBSERVATION_KWARGS}
            if extra:
                metadata = dict(kwargs.get("metadata") or {})
                metadata.update(extra)
                kwargs["metadata"] = metadata
            cm = self._client.start_as_current_observation(name=name, **kwargs)
            cm.__enter__()
        except Exception:
            cm = None
        try:
            yield
        finally:
            if cm is not None:
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    pass

    def log(self, **fields) -> None:
        try:
            usage = fields.pop("usage_details", None)
            cost = fields.pop("cost_details", None)
            output = fields.pop("output", None)
            level = fields.pop("level", None)
            status = fields.pop("status_message", None)
            metadata = dict(fields.pop("metadata", None) or {})
            if fields:  # leftover scalars (latency_ms, iterations, ...) -> metadata
                metadata.update(fields)
            metadata = metadata or None

            # usage/cost only make sense on a generation; everything else is a span.
            if usage is not None or cost is not None:
                self._client.update_current_generation(
                    output=output, usage_details=usage, cost_details=cost, metadata=metadata
                )
            else:
                self._client.update_current_span(
                    output=output, level=level, status_message=status, metadata=metadata
                )
        except Exception:
            pass

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            pass


_tracer: Tracer = NoopTracer()


def get_tracer() -> Tracer:
    """Return the active tracer (no-op until a real one is installed)."""
    return _tracer


def set_tracer(tracer: Tracer) -> None:
    """Install a tracer implementation."""
    global _tracer
    _tracer = tracer


def init_tracer() -> Tracer:
    """Install a Langfuse tracer if configured + reachable; else keep the no-op.

    Reads ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_HOST`` from
    the environment. Returns the active tracer either way, so callers can tell
    whether tracing is on (``isinstance(t, NoopTracer)``).
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not (public_key and secret_key):
        return get_tracer()

    try:
        from langfuse import Langfuse
    except ImportError:
        return get_tracer()  # 'obs' extra not installed

    try:
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=os.getenv("LANGFUSE_HOST") or None,
        )
        if not client.auth_check():
            return get_tracer()
    except Exception:
        return get_tracer()  # bad keys / network — degrade silently

    tracer = LangfuseTracer(client)
    set_tracer(tracer)
    return tracer


def flush() -> None:
    """Flush any buffered traces on the active tracer (call before exit)."""
    flusher = getattr(get_tracer(), "flush", None)
    if callable(flusher):
        flusher()
