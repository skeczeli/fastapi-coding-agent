"""Persistent per-project memory (#C6).

Knowledge that outlives a single run: detected architecture, important files,
conventions, decisions, session summaries. Backed by a directory of JSON files
(one per category) so entries are human-readable and diffable.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

_LIST_CATEGORIES = frozenset(
    {
        "important_files",
        "dependencies",
        "commands",
        "conventions",
        "decisions",
        "bugs",
        "session_summaries",
    }
)


def _atomic_write(path: str, content: str) -> None:
    """Write *content* to *path* atomically (temp file + rename)."""
    directory = os.path.dirname(path)
    if not directory:
        directory = "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        os.write(fd, content.encode())
        os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.close(fd)
        except Exception:
            pass
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


@dataclass
class ProjectMemory:
    """Per-project persistent memory backed by a ``.agent_memory/`` directory."""

    data: dict[str, Any] = field(default_factory=dict)
    path: str = ".agent_memory"

    @classmethod
    def load(cls, path: str = ".agent_memory") -> ProjectMemory:
        """Load memory from a directory of JSON files.

        Returns an empty ``ProjectMemory`` if the directory is missing.
        Skips individual files that fail to parse (logs a warning).
        """
        mem = cls(path=path)
        if not os.path.isdir(path):
            return mem
        for fname in os.listdir(path):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(path, fname)
            try:
                with open(fpath) as f:
                    raw = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("skipping corrupt memory file %s: %s", fpath, exc)
                continue
            key = fname.removesuffix(".json")
            if "summary" in raw:
                mem.data[key] = raw["summary"]
            elif "entries" in raw:
                mem.data[key] = list(raw["entries"])
            else:
                mem.data[key] = raw
        return mem

    def save(self, path: str | None = None) -> None:
        """Persist all categories to disk."""
        dest = path or self.path
        os.makedirs(dest, exist_ok=True)
        for key, value in self.data.items():
            self._write_category(key, value, dest)

    def _write_category(self, key: str, value: Any, dest: str | None = None) -> None:
        dest = dest or self.path
        os.makedirs(dest, exist_ok=True)
        fpath = os.path.join(dest, f"{key}.json")
        if isinstance(value, list):
            payload = {"entries": value}
        elif isinstance(value, str):
            payload = {"summary": value}
        else:
            payload = value
        _atomic_write(fpath, json.dumps(payload, indent=2, ensure_ascii=False))

    def remember(self, category: str, entry: str) -> None:
        """Append *entry* to a list-typed category (deduplicates). Persists immediately."""
        entries = self.data.setdefault(category, [])
        if not isinstance(entries, list):
            entries = [entries]
            self.data[category] = entries
        if entry not in entries:
            entries.append(entry)
        self._write_category(category, entries)

    def get_category(self, category: str) -> list[str]:
        """Return entries for *category* (empty list if missing or non-list)."""
        value = self.data.get(category, [])
        if isinstance(value, list):
            return list(value)
        return []

    def set_architecture(self, summary: str) -> None:
        """Overwrite the architecture summary. Persists immediately."""
        self.data["architecture"] = summary
        self._write_category("architecture", summary)

    def add_session_summary(self, summary: str) -> None:
        """Append a timestamped session summary. Persists immediately."""
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        entry = f"[{ts}] {summary}"
        self.remember("session_summaries", entry)
