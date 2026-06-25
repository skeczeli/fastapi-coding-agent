"""Tests for the rag_search tool (#B3 prerequisite)."""

import pytest

pytest.importorskip("chromadb")

from agent import tools  # noqa: E402
from agent.tools.rag import rag_search  # noqa: E402


def test_rag_search_is_registered():
    tool = tools.get("rag_search")
    assert tool.name == "rag_search"
    assert tool.permission == "read"


from agent.rag import store  # noqa: E402

_VECS = {
    "body": [1.0, 0.0, 0.0],
    "deploy": [0.0, 0.0, 1.0],
}

_CHUNKS = [
    ("tutorial/body.md", "Tutorial > Request Body", "Use a Pydantic model.", "body"),
    ("deployment/index.md", "", "Ship it with uvicorn.", "deploy"),
]


@pytest.fixture
def populated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(store, "embed_texts", lambda texts, **kw: [_VECS[t] for t in texts])
    collection = store.get_collection(reset=True)
    collection.upsert(
        ids=[f"{src}::{i}" for i, (src, *_) in enumerate(_CHUNKS)],
        documents=[doc for *_, doc, _ in _CHUNKS],
        embeddings=[_VECS[key] for *_, key in _CHUNKS],
        metadatas=[{"source": src, "section": sec} for src, sec, _, _ in _CHUNKS],
    )
    return collection


def test_rag_search_returns_formatted_results(populated_store, monkeypatch):
    monkeypatch.setattr(store, "embed_texts", lambda texts, **kw: [_VECS["body"]])
    result = rag_search.execute({"query": "body"})
    assert "[rag]" in result
    assert "tutorial/body.md" in result
    assert "Pydantic model" in result


def test_rag_search_empty_query():
    result = rag_search.execute({"query": "   "})
    assert "[error]" in result


def test_rag_search_no_results(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CHROMA_DIR", str(tmp_path / "empty"))
    store.get_collection(reset=True)
    result = rag_search.execute({"query": "anything"})
    assert "(no RAG results" in result
