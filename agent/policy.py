"""Policy engine — validates tool calls against agent.config.yaml (#A3).

Checks each call in order: deny patterns, workspace confinement,
require_approval. Returns None if allowed, or a reason string if denied.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from agent.config import Config
from agent.tools import Tool

# Process-wide approval handler for ``require_approval`` commands. ``check`` falls
# back to this when the caller doesn't pass an explicit ``approval_fn`` — so the
# CLI can wire one confirm-before-run prompt that reaches every loop (orchestrator
# and subagents), without threading the callback through each ``Subagent.run``.
# ``None`` (e.g. in tests) → such commands are denied instead of prompting.
_default_approval_fn: Callable[[str], bool] | None = None


def set_approval_fn(fn: Callable[[str], bool] | None) -> None:
    """Install the process-wide approval handler used by ``check`` (see above)."""
    global _default_approval_fn
    _default_approval_fn = fn


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a glob pattern (with ``**`` support) to a compiled regex."""
    i, n, regex = 0, len(pattern), ""
    while i < n:
        c = pattern[i]
        if c == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                regex += ".*"
                i += 2
                if i < n and pattern[i] == "/":
                    regex += "/?"
                    i += 1
            else:
                regex += "[^/]*"
                i += 1
        elif c == "?":
            regex += "[^/]"
            i += 1
        elif c in r"\.()[]{}+^$|":
            regex += "\\" + c
            i += 1
        else:
            regex += c
            i += 1
    return re.compile(f"^{regex}$")


def _path_matches_any(path: str, patterns: list[str]) -> str | None:
    """Return the first deny pattern that matches ``path``, or ``None``."""
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]

    for pattern in patterns:
        if "/" not in pattern and "**" not in pattern:
            basename = normalized.rsplit("/", 1)[-1] if "/" in normalized else normalized
            if _glob_to_regex(pattern).match(basename):
                return pattern
        else:
            if _glob_to_regex(pattern).match(normalized):
                return pattern
    return None


def _is_within_workspace(path_str: str, workspace: str) -> bool:
    """Check if the resolved path is inside the workspace directory."""
    resolved = Path(path_str).resolve()
    ws = Path(workspace).resolve()
    try:
        resolved.relative_to(ws)
        return True
    except ValueError:
        return False


def _command_matches_any(command: str, patterns: list[str]) -> str | None:
    """Return the first pattern matching any segment of ``command``.

    Splits the command by shell operators (``&&``, ``||``, ``;``, ``|``)
    and checks each segment independently, so chained commands like
    ``cd /tmp && rm -rf /`` are caught.

    For commands, ``*`` matches everything including ``/`` (unlike path globs).
    """
    segments = re.split(r"\s*(?:&&|\|\||;|\|)\s*", command)
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        for pattern in patterns:
            # Build regex where * matches everything (including /)
            regex_str = "^"
            i = 0
            while i < len(pattern):
                if pattern[i] == "*":
                    if i + 1 < len(pattern) and pattern[i + 1] == "*":
                        regex_str += ".*"
                        i += 2
                    else:
                        regex_str += ".*"  # For commands, * matches everything
                        i += 1
                elif pattern[i] == "?":
                    regex_str += "."
                    i += 1
                elif pattern[i] in r"\.()[]{}+^$|":
                    regex_str += "\\" + pattern[i]
                    i += 1
                else:
                    regex_str += pattern[i]
                    i += 1
            regex_str += "$"
            if re.match(regex_str, segment):
                return pattern
    return None


def check(
    tool: Tool,
    args: dict,
    config: Config,
    approval_fn: Callable[[str], bool] | None = None,
) -> str | None:
    """Validate a tool call against policy rules.

    Returns ``None`` if the call is allowed, or a human-readable reason
    string if it should be blocked.
    """
    if tool.permission == "read":
        path = args.get("path", "")
        if path and config.read.deny:
            matched = _path_matches_any(path, config.read.deny)
            if matched:
                return f"path matches read.deny pattern: {matched}"

    elif tool.permission == "write":
        path = args.get("path", "")
        if path:
            if config.write.deny:
                matched = _path_matches_any(path, config.write.deny)
                if matched:
                    return f"path matches write.deny pattern: {matched}"
            if not _is_within_workspace(path, config.workspace):
                return f"path is outside workspace: {config.workspace}"

    elif tool.permission == "command":
        command = args.get("command", "")
        if command:
            if config.commands.deny:
                matched = _command_matches_any(command, config.commands.deny)
                if matched:
                    return f"command matches commands.deny pattern: {matched}"
            if config.commands.require_approval:
                matched = _command_matches_any(
                    command, config.commands.require_approval
                )
                if matched:
                    # Use the explicit handler if given, else the process-wide default.
                    handler = approval_fn if approval_fn is not None else _default_approval_fn
                    if handler is None:
                        return (
                            f"command requires approval (no approval handler): {matched}"
                        )
                    if not handler(f"run command: {command}"):
                        return f"command rejected by user: {matched}"

    return None
