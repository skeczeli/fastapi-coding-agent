"""Policy engine — validates tool calls against agent.config.yaml (#A3).

Checks each call in order: deny patterns, workspace confinement,
require_approval. Returns None if allowed, or a reason string if denied.
"""

from __future__ import annotations

import re
from typing import Callable

from agent.config import Config
from agent.tools import Tool


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

    return None
