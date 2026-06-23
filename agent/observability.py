"""Observability — tracer interface + a no-op default.

The contract (#0) so other code can wrap LLM calls, tools, and retrievals with
spans/logs *before* Langfuse is wired. The real Langfuse-backed tracer is #B4
(Dev 2); until then ``get_tracer`` returns a ``NoopTracer`` that does nothing.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class Tracer(Protocol):
    """Minimal tracing surface other modules instrument against."""

    def span(self, name: str, **attrs) -> "Iterator[None]":
        """Context manager wrapping a unit of work (LLM call, tool, retrieval)."""
        ...

    def log(self, **fields) -> None:
        """Record a structured event (tokens, latency, model, errors...)."""
        ...


class NoopTracer:
    """Does nothing — the default until Langfuse lands in #B4."""

    @contextmanager
    def span(self, name: str, **attrs) -> Iterator[None]:
        yield

    def log(self, **fields) -> None:
        pass


_tracer: Tracer = NoopTracer()


def get_tracer() -> Tracer:
    """Return the active tracer (no-op until #B4 installs a real one)."""
    return _tracer


def set_tracer(tracer: Tracer) -> None:
    """Install a tracer implementation (called by #B4)."""
    global _tracer
    _tracer = tracer
