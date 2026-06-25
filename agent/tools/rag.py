"""RAG search tool — exposes the vector store to the harness (#B3).

Wraps ``retrieve()`` (#B2) as a tool the LLM can call. Results are formatted
as text for the LLM and returned; source recording into ``TaskState`` is
handled by the Researcher's wrapper, not here.
"""

from __future__ import annotations

from agent.rag.retrieve import retrieve
from agent.tools import Permission, register
from agent.tools.base import _BaseTool


def _rag_search(args: dict) -> str:
    query = args.get("query", "")
    k = args.get("k", 5)
    if not query.strip():
        return "[error] rag_search: 'query' is empty"

    sources = retrieve(query, k=k)
    if not sources:
        return f"(no RAG results for: {query})"

    lines: list[str] = []
    for s in sources:
        score_str = f" (score: {s.score:.2f})" if s.score is not None else ""
        lines.append(f"- [{s.origin}] {s.ref}{score_str}\n  {s.snippet[:300]}")
    return "\n".join(lines)


rag_search = _BaseTool(
    name="rag_search",
    description="Search the FastAPI documentation RAG store. Use this FIRST before web_search.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query."},
            "k": {"type": "integer", "description": "Number of results (default 5)."},
        },
        "required": ["query"],
    },
    permission="read",
    _impl=_rag_search,
)

register(rag_search)
