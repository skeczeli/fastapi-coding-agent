"""Researcher subagent (#B3).

RAG-first: queries the vector store via ``rag_search``; if evidence is
insufficient, falls back to ``web_search``. Records every source to shared
state via recording wrappers and labels each by origin (RAG / web).
"""

from __future__ import annotations

from dataclasses import dataclass

from agent import harness, tools
from agent.state import TaskState


RESEARCHER_PROMPT = """\
You are the Researcher in a multi-agent coding agent specialized in FastAPI.
Your job is to find authoritative information to answer the task.

Strategy — RAG first, web second:
1. ALWAYS start by calling ``rag_search`` with a well-crafted query.
2. If the RAG results are sufficient to answer confidently, respond immediately.
3. Only call ``web_search`` when RAG results are missing, ambiguous, or outdated.
   Prefer official FastAPI / Starlette / Pydantic documentation in web results.
4. You may call ``rag_search`` multiple times with rephrased queries.

Your answer must:
- Cite which sources you relied on (by reference).
- Label each piece of information by origin: [RAG], [web], or [inference] \
(for your own reasoning not grounded in a source).
- Be concise and actionable — the Implementer will use your findings to write code."""


@dataclass
class _RecordingRagSearch:
    """Wraps ``rag_search`` to record retrieved sources into ``state.sources``."""

    inner: tools.Tool
    state: TaskState

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def description(self) -> str:
        return self.inner.description

    @property
    def parameters(self) -> dict:
        return self.inner.parameters

    @property
    def permission(self) -> str:
        return self.inner.permission

    def execute(self, args: dict) -> str:
        from agent.rag import retrieve
        query = args.get("query", "")
        k = args.get("k", 5)
        sources = retrieve.retrieve(query, k=k)
        for s in sources:
            self.state.add_source(s.origin, s.ref, s.snippet, s.score)
        return self.inner.execute(args)


@dataclass
class _RecordingWebSearch:
    """Wraps ``web_search`` to record results as sources with origin='web'."""

    inner: tools.Tool
    state: TaskState

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def description(self) -> str:
        return self.inner.description

    @property
    def parameters(self) -> dict:
        return self.inner.parameters

    @property
    def permission(self) -> str:
        return self.inner.permission

    def execute(self, args: dict) -> str:
        result = self.inner.execute(args)
        query = args.get("query", "")
        if not result.startswith("[error]") and not result.startswith("(no web"):
            self.state.add_source("web", query, snippet=result[:200])
        return result


def _findings_preamble(state: TaskState) -> str:
    """Render prior subagent findings to ground the research."""
    parts: list[str] = []
    result = state.subagent_results.get("explorer")
    if result:
        parts.append(f"### explorer findings\n{result}")
    return "\n\n".join(parts)


class Researcher:
    name = "researcher"
    allowed_tools = ["rag_search", "web_search"]

    def _tools(self, state: TaskState) -> list[tools.Tool]:
        resolved: list[tools.Tool] = []
        for tool_name in self.allowed_tools:
            tool = tools.get(tool_name)
            if tool_name == "rag_search":
                tool = _RecordingRagSearch(inner=tool, state=state)
            elif tool_name == "web_search":
                tool = _RecordingWebSearch(inner=tool, state=state)
            resolved.append(tool)
        return resolved

    def run(self, state: TaskState, task: str) -> str:
        preamble = _findings_preamble(state)
        user_msg = f"{task}\n\n{preamble}" if preamble else task

        result = harness.run_loop(
            system_prompt=RESEARCHER_PROMPT,
            tool_list=self._tools(state),
            state=state,
            user_msg=user_msg,
        )
        state.subagent_results[self.name] = result
        return result
