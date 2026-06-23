"""Researcher subagent — stub (#0). Real impl: #B3 (Dev 3).

RAG-first (queries the vector store via #B2 retrieve), web-search fallback;
records sources to state and labels each by origin (RAG / web / inference).
"""

from __future__ import annotations

from agent.state import TaskState


class Researcher:
    name = "researcher"
    allowed_tools = ["rag_search", "web_search"]

    def run(self, state: TaskState, task: str) -> str:
        result = f"[researcher stub] would research (RAG first): {task}"
        state.subagent_results[self.name] = result
        return result
