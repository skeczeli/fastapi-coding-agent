"""RAG retrieval + source attribution — signature stub (#0), implemented in #B2.

CONTRACT: ``retrieve(query, k) -> list[Source]`` is consumed by the Researcher
subagent (#B3, Dev 3). The signature must not change without coordinating —
other lanes mock it until #B2 lands.
"""

from __future__ import annotations

from agent.state import Source


def retrieve(query: str, k: int = 5) -> list[Source]:
    """Return the top-k FastAPI doc chunks for ``query``, each with provenance.

    Embeds the query, searches the vector store, and returns ``Source`` objects
    (``origin="rag"``, ``ref=file/section``, ``snippet``, ``score``).

    Stub for #0 — returns an empty list. Implemented in #B2.
    """
    return []
