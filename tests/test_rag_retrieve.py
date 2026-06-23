"""Tests for RAG retrieval (#B2).

These never hit OpenAI: ``store.embed_texts`` is monkeypatched with a tiny
deterministic fake, and the chunks are upserted into a throwaway Chroma DB in
``tmp_path``. So we exercise the real query path + ``Source`` mapping offline.
Skips if Chroma isn't installed (mirrors the keyless smoke tests).
"""

import pytest

pytest.importorskip("chromadb")

from agent.rag import retrieve, store  # noqa: E402
from agent.state import Source  # noqa: E402

# Three orthogonal 3-d vectors so similarity is unambiguous: a query equal to one
# doc's vector retrieves that doc first.
_VECS = {
    "body": [1.0, 0.0, 0.0],
    "query params": [0.0, 1.0, 0.0],
    "deploy": [0.0, 0.0, 1.0],
}

_CHUNKS = [
    ("tutorial/body.md", "Tutorial > Request Body", "Use a Pydantic model.", "body"),
    ("tutorial/query.md", "Tutorial > Query Params", "Declare query parameters.", "query params"),
    ("deployment/index.md", "", "Ship it with uvicorn.", "deploy"),
]


@pytest.fixture
def populated_store(tmp_path, monkeypatch):
    """A persisted Chroma collection filled with the fixture chunks."""
    monkeypatch.setenv("AGENT_CHROMA_DIR", str(tmp_path / "chroma"))
    # Fake embedder: look up the canned vector by exact text match.
    monkeypatch.setattr(store, "embed_texts", lambda texts, **kw: [_VECS[t] for t in texts])

    collection = store.get_collection(reset=True)
    collection.upsert(
        ids=[f"{src}::{i}" for i, (src, *_) in enumerate(_CHUNKS)],
        documents=[doc for *_, doc, _ in _CHUNKS],
        embeddings=[_VECS[key] for *_, key in _CHUNKS],
        metadatas=[{"source": src, "section": sec} for src, sec, _, _ in _CHUNKS],
    )
    return collection


def test_retrieve_returns_sources_best_first(populated_store, monkeypatch):
    # Query embeds to the "body" vector -> that chunk must rank first.
    monkeypatch.setattr(store, "embed_texts", lambda texts, **kw: [_VECS["body"]])

    results = retrieve.retrieve("how do I declare a request body?", k=3)

    assert len(results) == 3
    assert all(isinstance(s, Source) and s.origin == "rag" for s in results)

    top = results[0]
    assert top.ref == "tutorial/body.md > Tutorial > Request Body"
    assert top.snippet == "Use a Pydantic model."
    assert top.score == pytest.approx(1.0)  # identical vector -> similarity 1
    # Scores are sorted best-first and stay within [0, 1].
    scores = [s.score for s in results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_ref_omits_breadcrumb_when_absent(populated_store, monkeypatch):
    monkeypatch.setattr(store, "embed_texts", lambda texts, **kw: [_VECS["deploy"]])
    top = retrieve.retrieve("how to deploy", k=1)[0]
    assert top.ref == "deployment/index.md"  # no " > " section suffix


def test_k_is_capped_at_collection_size(populated_store, monkeypatch):
    monkeypatch.setattr(store, "embed_texts", lambda texts, **kw: [_VECS["body"]])
    # Asking for more than we have returns everything, not an error.
    assert len(retrieve.retrieve("anything", k=99)) == 3


def test_blank_query_returns_empty(populated_store):
    assert retrieve.retrieve("   ") == []


def test_empty_store_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_CHROMA_DIR", str(tmp_path / "empty"))
    store.get_collection(reset=True)  # exists but unpopulated
    assert retrieve.retrieve("anything") == []
