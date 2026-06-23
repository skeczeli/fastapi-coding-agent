"""RAG ingestion pipeline — stub (#0), implemented in #B1.

Fetches FastAPI's official docs (markdown from tiangolo/fastapi/docs), chunks
them, embeds with OpenAI embeddings, and stores in a vector DB (Chroma).
Entry point: ``python -m agent.rag.ingest``.
"""

from __future__ import annotations


def main() -> None:
    """Build/refresh the vector store. Stub for #0 — implemented in #B1."""
    print("[ingest stub] RAG ingestion is implemented in #B1.")


if __name__ == "__main__":
    main()
