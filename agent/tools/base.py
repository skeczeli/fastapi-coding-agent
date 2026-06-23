"""Base tools — stubs only (#0).

The five base tools every agent shares. Here they declare their name, schema,
and permission category and self-register, but ``execute`` raises
``NotImplementedError`` — the real bodies land in #A2 (Dev 2). They exist now so
the registry is populated and imports/schemas resolve for the other lanes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agent.tools import Permission, register


@dataclass
class _BaseTool:
    """Minimal Tool stub. Concrete tools fill ``execute`` in #A2."""

    name: str
    description: str
    parameters: dict
    permission: Permission
    _impl: object = field(default=None, repr=False)

    def execute(self, args: dict) -> str:
        raise NotImplementedError(f"{self.name}: implemented in #A2")


read_file = _BaseTool(
    name="read_file",
    description="Read a file from the workspace and return its contents.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    permission="read",
)

write_file = _BaseTool(
    name="write_file",
    description="Create or overwrite a file in the workspace.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
    permission="write",
)

run_command = _BaseTool(
    name="run_command",
    description="Run a shell command in the workspace and return its output.",
    parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
    permission="command",
)

list_files = _BaseTool(
    name="list_files",
    description="List files/directories under a workspace path.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": [],
    },
    permission="read",
)

web_search = _BaseTool(
    name="web_search",
    description="Search the web (Tavily) and return relevant results. Fallback after RAG.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    permission="read",
)

# Self-register so the registry is populated on import.
for _t in (read_file, write_file, run_command, list_files, web_search):
    register(_t)
