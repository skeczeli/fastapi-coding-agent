"""Base tools — the five tools every agent shares (#A2).

``read_file``, ``write_file``, ``run_command``, ``list_files`` and ``web_search``.
Each declares its name, JSON-schema parameters and permission category, wires a
concrete implementation, and self-registers so the harness can hand its schema to
the LLM and dispatch calls by name.

Design notes:
- **No policy checks here.** Validating a call against ``agent.config.yaml``
  (deny globs, workspace confinement, require-approval) is the harness's job via
  the #A3 policy engine, *before* ``execute`` runs. Tools just do the work; the
  separation keeps guardrail logic in one place.
- **Errors are returned, not raised.** A tool's result is fed back to the LLM as
  text, so a failure comes back as an ``"[error] ..."`` string the model can read
  and react to. Raising would crash the harness loop instead.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agent.tools import Permission, register

# Cap so a hung command (e.g. a server that never exits) can't block the agent.
_COMMAND_TIMEOUT_S = 120


@dataclass
class _BaseTool:
    """A base tool: metadata + a concrete ``_impl`` that does the work.

    ``execute`` just forwards validated args to ``_impl``; keeping the body in a
    plain function (instead of a method per subclass) makes each tool a small,
    independently testable unit.
    """

    name: str
    description: str
    parameters: dict
    permission: Permission
    _impl: Callable[[dict], str] = field(repr=False)

    def execute(self, args: dict) -> str:
        return self._impl(args)


def _read_file(args: dict) -> str:
    """Return the text contents of ``path`` (or an ``[error]`` string)."""
    path = args.get("path", "")
    try:
        return Path(path).read_text()
    except FileNotFoundError:
        return f"[error] file not found: {path}"
    except IsADirectoryError:
        return f"[error] is a directory, not a file: {path}"
    except UnicodeDecodeError:
        return f"[error] not a UTF-8 text file (binary?): {path}"
    except OSError as exc:
        return f"[error] could not read {path}: {exc}"


def _write_file(args: dict) -> str:
    """Create or overwrite ``path`` with ``content``, making parent dirs."""
    path = args.get("path", "")
    content = args.get("content", "")
    if not path:
        return "[error] write_file: 'path' is required"
    try:
        p = Path(path)
        if p.parent and not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"[ok] wrote {len(content)} chars to {path}"
    except OSError as exc:
        return f"[error] could not write {path}: {exc}"


def _run_command(args: dict) -> str:
    """Run ``command`` in a shell; return exit code plus stdout/stderr."""
    command = args.get("command", "")
    if not command.strip():
        return "[error] run_command: 'command' is empty"
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return f"[error] command timed out after {_COMMAND_TIMEOUT_S}s: {command}"

    parts: list[str] = []
    if proc.stdout:
        parts.append(proc.stdout.rstrip())
    if proc.stderr:
        parts.append(f"[stderr]\n{proc.stderr.rstrip()}")
    body = "\n".join(parts) or "(no output)"
    return f"[exit {proc.returncode}]\n{body}"


def _list_files(args: dict) -> str:
    """List entries under ``path`` (defaults to "."), dirs marked with a trailing /."""
    path = args.get("path") or "."
    try:
        p = Path(path)
        if not p.exists():
            return f"[error] path not found: {path}"
        if p.is_file():
            return path
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        if not entries:
            return f"(empty directory: {path})"
        return "\n".join(f"{e.name}/" if e.is_dir() else e.name for e in entries)
    except OSError as exc:
        return f"[error] could not list {path}: {exc}"


def _web_search(args: dict) -> str:
    """Search the web via Tavily; the RAG fallback (#B3 calls this last).

    Needs the optional ``web`` extra (``tavily-python``) and ``TAVILY_API_KEY``.
    The key is read from the environment — entrypoints load ``.env`` before any
    tool runs (``llm`` does it on the first model call, which always precedes a
    tool call), so we don't re-load it here.
    """
    query = args.get("query", "")
    if not query.strip():
        return "[error] web_search: 'query' is empty"

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "[error] web_search unavailable: TAVILY_API_KEY not set"
    try:
        from tavily import TavilyClient
    except ImportError:
        return "[error] web_search unavailable: install the 'web' extra (pip install -e '.[web]')"

    try:
        client = TavilyClient(api_key=api_key)
        resp = client.search(query=query, max_results=5)
    except Exception as exc:  # network/auth/quota — surface, don't crash the loop
        return f"[error] web search failed: {exc}"

    results = resp.get("results", []) if isinstance(resp, dict) else []
    if not results:
        return f"(no web results for: {query})"
    lines = []
    for r in results:
        title = r.get("title", "").strip()
        url = r.get("url", "").strip()
        snippet = (r.get("content", "") or "").strip()[:300]
        lines.append(f"- {title}\n  {url}\n  {snippet}")
    return "\n".join(lines)


read_file = _BaseTool(
    name="read_file",
    description="Read a file from the workspace and return its contents.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    permission="read",
    _impl=_read_file,
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
    _impl=_write_file,
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
    _impl=_run_command,
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
    _impl=_list_files,
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
    _impl=_web_search,
)

# Self-register so the registry is populated on import.
for _t in (read_file, write_file, run_command, list_files, web_search):
    register(_t)
