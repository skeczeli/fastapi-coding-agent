# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo.

## Project

Multi-agent AI coding agent **specialized in FastAPI**, evolved from an in-class single-agent harness. Academic final project (TP Final IA). See `README.md` for the overview and `BACKLOG.md` for the full work breakdown and issue ownership.

## Hard constraints (from the assignment — do not violate)

- **No agent-orchestration frameworks.** No LangChain, LangGraph, CrewAI, AutoGen, or similar. The harness and agent coordination are hand-written. Point libraries for embeddings, vector storage, web search, observability, and CLI are allowed.
- **Build on the in-class harness.** Keep the base harness and base tools (read/write/run/list/web_search); evolve, don't discard. The base agent this is built on lives in the in-class TP notebook: https://colab.research.google.com/drive/1m47p4bq8EEAD16-tXPDld0nnHBUU5ilg?usp=sharing
- **Every tool call is validated** against `agent.config.yaml` policies before execution.
- **RAG first, web second.** Consult the RAG over FastAPI docs before deciding; fall back to web search only when evidence is insufficient. Always surface which sources were used and label info origin (repo / memory / RAG / web / inference).

## Stack

- Python + OpenAI SDK. Vector store (Chroma or FAISS) + OpenAI embeddings. Tavily for web search. Langfuse for observability.

## Architecture (target)

- **Orchestrator** — main agent; owns the shared `TaskState`; coordinates subagents via the *agents-as-tools* pattern (its "tools" are the subagents). It delegates rather than acting directly; calling base tools itself is a future extension (#C1 ships delegation-only, since base tools enforce permission per-subagent).
- **Subagents** — Explorer (understands the repo), Researcher (RAG + web), Implementer (writes code), Tester (runs checks), Reviewer (validates the diff vs. the request). Each is its own harness loop with a specific prompt and a subset of tools/permissions.
- **The harness loop is the shared primitive** — one reusable `run_loop(prompt, tools, state)` used by the orchestrator *and* every subagent. Defined in #0, hardened in #A1.
- **Shared state** (`TaskState`): original request, plan, progress, per-subagent results, sources consulted, files modified, observations.
- **Persistent memory**: per-project knowledge that outlives a single conversation.

## Conventions

- **Contract-first.** Interfaces live in #0 (`Tool`, `Subagent`, `TaskState`, `llm.complete`, tracer). Code against the interface and stub/mock cross-lane dependencies — don't wait on another lane's implementation.
- **One place calls the LLM:** `agent/llm.py` (with a mock mode for tests that don't hit the API).
- **Tools self-register** with the registry and declare a permission category (read / write / command).
- Type hints + docstrings on public interfaces. Tests alongside new modules.

## Commits & PRs

- Branch off `main`; keep PRs scoped to a single issue where possible. Reference the issue by its backlog ID (e.g. `#A1`) and/or GitHub number.
- **Small commits — one unit of change each.** Commit each logical step on its own (e.g. "scaffold packaging", "add TaskState contract", "add tool registry") rather than batching unrelated changes. Each commit should build/import cleanly on its own.

## Where to start

Issue **#0 (foundation + contracts)** blocks everything — it must be merged before parallel work begins.
