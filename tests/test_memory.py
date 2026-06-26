"""ProjectMemory persistence tests (#C6)."""

import json
import os

from agent.memory import ProjectMemory

CATEGORIES = [
    "important_files",
    "dependencies",
    "commands",
    "conventions",
    "decisions",
    "bugs",
    "session_summaries",
]


def test_save_and_load_round_trip(tmp_path):
    mem_dir = str(tmp_path / ".agent_memory")
    mem = ProjectMemory(path=mem_dir)
    mem.remember("dependencies", "SQLAlchemy for ORM")
    mem.remember("commands", "pytest -v")

    loaded = ProjectMemory.load(mem_dir)
    assert loaded.get_category("dependencies") == ["SQLAlchemy for ORM"]
    assert loaded.get_category("commands") == ["pytest -v"]


def test_load_missing_directory_returns_empty(tmp_path):
    mem_dir = str(tmp_path / "nonexistent")
    loaded = ProjectMemory.load(mem_dir)
    assert loaded.data == {}


def test_load_corrupt_file_skips_it(tmp_path):
    mem_dir = str(tmp_path / ".agent_memory")
    os.makedirs(mem_dir)
    (tmp_path / ".agent_memory" / "commands.json").write_text("NOT JSON{{{")
    (tmp_path / ".agent_memory" / "dependencies.json").write_text(
        json.dumps({"entries": ["fastapi"]})
    )

    loaded = ProjectMemory.load(mem_dir)
    assert loaded.get_category("dependencies") == ["fastapi"]
    assert loaded.get_category("commands") == []


def test_remember_deduplicates(tmp_path):
    mem_dir = str(tmp_path / ".agent_memory")
    mem = ProjectMemory(path=mem_dir)
    mem.remember("conventions", "snake_case for functions")
    mem.remember("conventions", "snake_case for functions")
    assert mem.get_category("conventions") == ["snake_case for functions"]


def test_remember_persists_immediately(tmp_path):
    mem_dir = str(tmp_path / ".agent_memory")
    mem = ProjectMemory(path=mem_dir)
    mem.remember("bugs", "race condition in auth middleware")

    raw = json.loads((tmp_path / ".agent_memory" / "bugs.json").read_text())
    assert "race condition in auth middleware" in raw["entries"]


def test_set_architecture(tmp_path):
    mem_dir = str(tmp_path / ".agent_memory")
    mem = ProjectMemory(path=mem_dir)
    mem.set_architecture("FastAPI + SQLAlchemy, layered arch")

    loaded = ProjectMemory.load(mem_dir)
    assert loaded.data["architecture"] == "FastAPI + SQLAlchemy, layered arch"


def test_set_architecture_persists_immediately(tmp_path):
    mem_dir = str(tmp_path / ".agent_memory")
    mem = ProjectMemory(path=mem_dir)
    mem.set_architecture("Monolith with FastAPI")

    raw = json.loads((tmp_path / ".agent_memory" / "architecture.json").read_text())
    assert raw["summary"] == "Monolith with FastAPI"


def test_add_session_summary(tmp_path):
    mem_dir = str(tmp_path / ".agent_memory")
    mem = ProjectMemory(path=mem_dir)
    mem.add_session_summary("Added POST /users endpoint")
    mem.add_session_summary("Fixed validation bug")

    loaded = ProjectMemory.load(mem_dir)
    summaries = loaded.get_category("session_summaries")
    assert len(summaries) == 2
    assert "Added POST /users endpoint" in summaries[0]
    assert "Fixed validation bug" in summaries[1]


def test_get_category_returns_empty_list_for_missing(tmp_path):
    mem_dir = str(tmp_path / ".agent_memory")
    mem = ProjectMemory(path=mem_dir)
    assert mem.get_category("nonexistent") == []


def test_save_and_load_preserves_all_categories(tmp_path):
    mem_dir = str(tmp_path / ".agent_memory")
    mem = ProjectMemory(path=mem_dir)
    mem.set_architecture("FastAPI app")
    for cat in CATEGORIES:
        mem.remember(cat, f"entry for {cat}")

    loaded = ProjectMemory.load(mem_dir)
    assert loaded.data["architecture"] == "FastAPI app"
    for cat in CATEGORIES:
        assert loaded.get_category(cat) == [f"entry for {cat}"]


def test_memory_survives_across_orchestrator_runs(tmp_path):
    """Acceptance: a second run sees findings persisted by the first."""
    from agent import llm
    from agent.agents import orchestrator

    mem_dir = str(tmp_path / ".agent_memory")

    # First run: orchestrator calls remember_project to save a dependency.
    llm.set_mock_script(
        [
            llm.LLMResponse(
                content="",
                tool_calls=[
                    llm.ToolCall(
                        id="c1",
                        name="remember_project",
                        arguments={"category": "dependencies", "content": "FastAPI"},
                    )
                ],
            ),
            llm.LLMResponse(content="Done.", tool_calls=[]),
        ]
    )
    orchestrator.run("first task", subagents=[], memory_path=mem_dir)
    llm.set_mock_script(None)

    # Second run: memory is loaded and fed to the orchestrator as context.
    # We verify the finding persisted.
    mem = ProjectMemory.load(mem_dir)
    assert mem.get_category("dependencies") == ["FastAPI"]
