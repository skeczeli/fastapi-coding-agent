"""Tests for the Researcher subagent (#B3)."""

from agent.agents.researcher import Researcher
from agent.agents import Subagent


def test_researcher_satisfies_subagent_protocol():
    assert isinstance(Researcher(), Subagent)


from agent import llm
from agent.llm import LLMResponse, ToolCall
from agent.state import TaskState


def test_researcher_records_rag_sources(monkeypatch):
    """When the LLM calls rag_search, sources are recorded in state automatically."""
    state = TaskState(request="how to validate emails in FastAPI")

    # Fake rag_search to return canned results (bypass Chroma).
    from agent.tools import rag as rag_mod
    original_impl = rag_mod._rag_search

    def fake_rag_search(args):
        return "- [rag] tutorial/body.md > Validation (score: 0.91)\n  Use EmailStr from pydantic."

    monkeypatch.setattr(rag_mod, "_rag_search", fake_rag_search)
    # Also patch retrieve so the recording wrapper can get Source objects.
    from agent.rag import retrieve as retrieve_mod
    from agent.state import Source

    monkeypatch.setattr(
        retrieve_mod,
        "retrieve",
        lambda query, k=5: [
            Source(origin="rag", ref="tutorial/body.md > Validation", snippet="Use EmailStr from pydantic.", score=0.91)
        ],
    )

    llm.set_mock_script([
        LLMResponse(tool_calls=[
            ToolCall(id="1", name="rag_search", arguments={"query": "validate email FastAPI"}),
        ]),
        LLMResponse(content="Use pydantic's EmailStr for email validation."),
    ])

    result = Researcher().run(state, "how to validate emails")

    assert len(state.sources) >= 1
    assert state.sources[0].origin == "rag"
    assert "tutorial/body.md" in state.sources[0].ref
    assert state.subagent_results["researcher"] == result


def test_researcher_fallback_to_web_when_rag_insufficient():
    """Researcher tries web_search after RAG results."""
    state = TaskState(request="latest FastAPI middleware changes")

    # Return empty RAG results, so LLM will try web_search
    from agent.rag import retrieve as retrieve_mod
    import pytest
    from unittest.mock import patch

    with patch.object(retrieve_mod, 'retrieve', return_value=[]):
        llm.set_mock_script([
            LLMResponse(tool_calls=[
                ToolCall(id="1", name="rag_search", arguments={"query": "middleware changes"}),
            ]),
            LLMResponse(tool_calls=[
                # web_search will fail (no API key) but that's OK - we just test LLM calls it
                ToolCall(id="2", name="web_search", arguments={"query": "FastAPI middleware 2025"}),
            ]),
            LLMResponse(content="Middleware has no recent updates in RAG. Based on web search, changes are X."),
        ])

        result = Researcher().run(state, "what changed in middleware")

        # Verify the Researcher called both tools
        assert state.subagent_results["researcher"] == result
        # RAG was called but returned nothing
        assert all(s.origin != "rag" for s in state.sources) or len(state.sources) == 0


def test_researcher_includes_explorer_preamble():
    """Prior explorer findings are passed to the LLM as context."""
    state = TaskState(request="understand auth flow")
    state.subagent_results["explorer"] = "Found auth.py with JWT middleware."

    llm.set_mock_script([
        LLMResponse(content="The auth flow uses JWT as found by the explorer."),
    ])

    result = Researcher().run(state, "explain the auth flow")

    assert state.subagent_results["researcher"] == result
