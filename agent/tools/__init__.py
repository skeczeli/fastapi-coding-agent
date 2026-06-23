"""Tool interface + registry.

A ``Tool`` is anything the LLM can invoke (read a file, run a command, search
the web, query RAG...). Tools self-register here so the harness can look them up
by name and hand their schemas to the LLM in OpenAI tool-calling format.

The interface is a ``Protocol`` (duck-typed) — a tool just needs the right
attributes/methods, no mandatory base class. Implementations live in #A2
(base tools) and #B2 (rag_search).
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

Permission = Literal["read", "write", "command"]


@runtime_checkable
class Tool(Protocol):
    """Something the agent can call.

    Attributes:
        name: Unique identifier the LLM uses to call the tool.
        description: What it does (shown to the LLM).
        parameters: JSON Schema describing the tool's arguments.
        permission: Category checked against config before execution.
    """

    name: str
    description: str
    parameters: dict
    permission: Permission

    def execute(self, args: dict) -> str:
        """Run the tool with validated args and return a string result."""
        ...


_REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> None:
    """Add a tool to the global registry (raises on duplicate names)."""
    if tool.name in _REGISTRY:
        raise ValueError(f"tool already registered: {tool.name!r}")
    _REGISTRY[tool.name] = tool


def get(name: str) -> Tool:
    """Look up a registered tool by name (raises ``KeyError`` if missing)."""
    return _REGISTRY[name]


def all_tools() -> list[Tool]:
    """Return every registered tool."""
    return list(_REGISTRY.values())


def _schema(t: Tool) -> dict:
    """One tool's OpenAI tool-calling schema."""
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        },
    }


def schemas(names: list[str] | None = None) -> list[dict]:
    """Return tool schemas in OpenAI tool-calling format.

    Args:
        names: Restrict to these tool names (e.g. a subagent's allowed subset).
            ``None`` returns all registered tools.
    """
    tools = all_tools() if names is None else [get(n) for n in names]
    return [_schema(t) for t in tools]


def schemas_for(tool_list: list[Tool]) -> list[dict]:
    """Build schemas straight from tool *objects*, bypassing the registry.

    The harness/orchestrator hand a loop its exact tool set (a subagent's
    allowed subset, or per-run subagent-as-tool adapters). Resolving those by
    name through the global registry would force every transient tool to be
    registered; building from the objects keeps each loop's toolset local.
    """
    return [_schema(t) for t in tool_list]


# Import base tools at the bottom (after register/get exist) so they self-register
# whenever the tools package is imported. Kept last to avoid a circular import.
from agent.tools import base as base  # noqa: E402,F401
