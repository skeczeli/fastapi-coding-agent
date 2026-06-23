"""Config loading — turns ``agent.config.yaml`` into a typed object.

Every tool call is validated against this config before it runs (assignment hard
constraint). #0 defines the shape and a loader; the actual policy *engine*
(glob matching, require-approval prompts, workspace confinement) lands in #A3.

TP1 lesson baked into the contract: don't rely on string-matching ``rm -rf`` —
also confine writes/deletes to ``workspace`` so guardrails can't be bypassed
(e.g. via ``shutil.rmtree``). ``workspace`` is here so #A3 can enforce that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PolicyConfig:
    """Deny/approval rules for one permission category (read/write/commands)."""

    deny: list[str] = field(default_factory=list)
    require_approval: list[str] = field(default_factory=list)


@dataclass
class Config:
    """Parsed agent configuration.

    Attributes:
        workspace: Root dir the agent is confined to for writes/commands.
        read: Read-permission policy.
        write: Write-permission policy.
        commands: Command-execution policy.
        raw: The untouched parsed YAML, for fields #A3 may add later.
    """

    workspace: str = "."
    read: PolicyConfig = field(default_factory=PolicyConfig)
    write: PolicyConfig = field(default_factory=PolicyConfig)
    commands: PolicyConfig = field(default_factory=PolicyConfig)
    raw: dict[str, Any] = field(default_factory=dict)


def _policy(section: dict[str, Any] | None) -> PolicyConfig:
    section = section or {}
    return PolicyConfig(
        deny=list(section.get("deny", [])),
        require_approval=list(section.get("require_approval", [])),
    )


def load_config(path: str = "config/agent.config.yaml") -> Config:
    """Load and parse the agent config file into a ``Config``.

    Args:
        path: Path to the YAML config.

    Returns:
        A typed ``Config``. Missing file/sections fall back to permissive
        defaults so tests and stubs work without a full config.
    """
    import yaml  # lazy: keeps importing this module cheap

    p = Path(path)
    data: dict[str, Any] = {}
    if p.exists():
        data = yaml.safe_load(p.read_text()) or {}

    return Config(
        workspace=data.get("workspace", "."),
        read=_policy(data.get("read")),
        write=_policy(data.get("write")),
        commands=_policy(data.get("commands")),
        raw=data,
    )
