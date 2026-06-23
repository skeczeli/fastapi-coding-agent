"""Shared test fixtures."""

import os

os.environ.setdefault("AGENT_LLM_MOCK", "1")  # never hit the real API in tests

import pytest

from agent import llm, tools


@pytest.fixture(autouse=True)
def isolate_tool_registry():
    """Snapshot the tool registry and restore it after each test.

    Tests that register temporary tools (e.g. a dummy "echo") shouldn't leak
    into others — keeps tests order-independent and registry-count-safe.
    """
    snapshot = dict(tools._REGISTRY)
    try:
        yield
    finally:
        tools._REGISTRY.clear()
        tools._REGISTRY.update(snapshot)


@pytest.fixture(autouse=True)
def clear_mock_script():
    """Clear any queued LLM mock script after each test.

    Same global-state hazard as the registry: a leftover script would feed
    canned responses into unrelated tests.
    """
    try:
        yield
    finally:
        llm.set_mock_script(None)
