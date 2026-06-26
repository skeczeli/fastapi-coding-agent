"""Shared vector-store access for the RAG lane (#B1 ingest + #B2 retrieve).

One place owns the store wiring so ingestion and retrieval agree on the
collection name, embedding model, and on-disk location. Both must embed queries
and documents with the *same* model, so that lives here too.

Choices (documented for #E3):
- **Vector store: Chroma** (``PersistentClient``) — zero-setup, persists to disk,
  handles metadata + similarity search out of the box. FAISS would mean managing
  the id↔metadata mapping by hand; not worth it for a single-machine project.
- **Embeddings: OpenAI ``text-embedding-3-small``** — cheap, 1536-dim, strong on
  short doc chunks (the ticket asks for it explicitly).
- **Distance: cosine** — the usual default for OpenAI embeddings.
- **We embed explicitly** via ``embed_texts`` (instead of Chroma's built-in
  embedding function) so #B2 reuses the exact same call for queries.
"""

from __future__ import annotations

import os

# On-disk location of the persisted Chroma DB (gitignored). Env-overridable so
# tests can point at a throwaway dir.
CHROMA_DIR = os.getenv("AGENT_CHROMA_DIR", "chroma_db")
COLLECTION_NAME = "fastapi_docs"

# The embedding model now comes from ``providers.embed_config()`` so the backend
# is env-selectable (OpenAI ``text-embedding-3-small`` by default; Gemini, etc.
# via AGENT_EMBED_*). Switching it changes the vector space — re-ingest after.
# OpenAI's embeddings endpoint accepts many inputs per call; batch to cut
# round-trips and stay well under request limits.
EMBED_BATCH = 100

_client = None


def _embed_client():
    """Lazily build the (OpenAI-compatible) embedding client for the configured provider.

    RAG-local on purpose — chat goes through ``llm.py``; embeddings can run on a
    different backend, so each owns its own client.
    """
    global _client
    if _client is None:
        from dotenv import load_dotenv  # ensure AGENT_EMBED_* / keys are loaded

        load_dotenv()
        from openai import OpenAI

        from agent.providers import embed_config

        _client = OpenAI(**embed_config().client_kwargs())
    return _client


def embed_texts(texts: list[str], model: str | None = None) -> list[list[float]]:
    """Embed a list of texts with the configured provider, batching under ``EMBED_BATCH``.

    Used for both documents (ingest) and queries (retrieve) so the vectors are
    comparable. ``model`` defaults to the configured embedding model. Returns one
    vector per input, in the same order.
    """
    if not texts:
        return []
    from agent.providers import embed_config

    model = model or embed_config().model
    client = _embed_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        resp = client.embeddings.create(model=model, input=batch)
        out.extend(d.embedding for d in resp.data)
    return out


def get_collection(*, persist_dir: str | None = None, reset: bool = False):
    """Return the persisted FastAPI-docs collection (creating it if needed).

    Args:
        persist_dir: Override the on-disk location (defaults to ``CHROMA_DIR``).
        reset: Drop and recreate the collection first (used by ``ingest --rebuild``).
    """
    import chromadb  # lazy: keeps importing this module cheap and key-free

    client = chromadb.PersistentClient(path=persist_dir or CHROMA_DIR)
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # didn't exist yet — fine
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
