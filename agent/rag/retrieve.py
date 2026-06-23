"""RAG retrieval + source attribution (#B2).

CONTRACT: ``retrieve(query, k) -> list[Source]`` is consumed by the Researcher
subagent (#B3, Dev 3). The signature must not change without coordinating —
other lanes mock it until #B2 lands.

The query is embedded with the *same* OpenAI model used at ingest time (via
``store.embed_texts``) so query and document vectors are comparable, then matched
against the persisted Chroma collection. Each hit comes back as a ``Source`` with
``origin="rag"`` so the orchestrator can label where the evidence came from.
"""

from __future__ import annotations

from agent.rag import store
from agent.state import Source


def _ref(meta: dict) -> str:
    """Build a human-readable provenance pointer from a chunk's metadata.

    ``"tutorial/body.md > Tutorial > Request Body"`` when a heading breadcrumb is
    present, otherwise just the file path.
    """
    source = meta.get("source", "?")
    section = meta.get("section") or ""
    return f"{source} > {section}" if section else source


def retrieve(query: str, k: int = 5) -> list[Source]:
    """Return the top-k FastAPI doc chunks for ``query``, each with provenance.

    Embeds the query, runs a cosine similarity search over the persisted store,
    and returns ``Source`` objects (``origin="rag"``, ``ref=file > section``,
    ``snippet=chunk text``, ``score=cosine similarity in [0, 1]``), best first.

    Returns an empty list for a blank query or an empty/missing store, so callers
    can treat "no evidence" uniformly (the "RAG first, web second" fallback).
    """
    if not query.strip():
        return []

    collection = store.get_collection()
    if collection.count() == 0:
        return []

    k = min(k, collection.count())
    query_embedding = store.embed_texts([query])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma nests each field one level deep (one list per query); we sent one.
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    sources: list[Source] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        # cosine distance -> similarity; clamp tiny negatives from float error.
        score = max(0.0, 1.0 - dist)
        sources.append(
            Source(origin="rag", ref=_ref(meta), snippet=doc, score=score)
        )
    return sources
