# FastAPI Coding Agent

> 🚧 **Work in progress.** This is the initial summary, not the final deliverable README. Install/config/run instructions and full documentation land in issue **#E3** once the system is integrated.

A multi-agent AI coding agent **specialized in FastAPI**, built on top of the harness from the in-class TP — **no orchestration frameworks** (no LangChain / LangGraph / CrewAI / AutoGen).

## What it is

The in-class TP was a single agent: a harness loop (LLM ↔ tools) with plan mode, supervision, and guardrails. This final project evolves it into a **team of specialized agents** coordinated by a main orchestrator, grounded in real FastAPI documentation via RAG.

## What it adds over the in-class TP

- **Multi-agent architecture** — an orchestrator that coordinates 5 subagents (Explorer, Researcher, Implementer, Tester, Reviewer) over a shared task state.
- **RAG over FastAPI docs** — chunking, embeddings, vector store; retrieves grounded context (with source attribution) before deciding. RAG first, web search as fallback.
- **Persistent project memory** — architecture, conventions, decisions, and session summaries that survive across runs.
- **Smart context handling** — summarizes history, detects no-progress loops, and knows when to stop or ask for help.
- **Config-driven safety** — `agent.config.yaml` policies validated before every tool call.
- **Observability** — full tracing (Langfuse): prompts, LLM calls, tools, retrieved docs, tokens, cost, latency.

## Use case

Specialization domain: **FastAPI**. The concrete, verifiable goal is defined in issue **#E1** (e.g. "add an authenticated endpoint with a Pydantic model; success = tests pass and the endpoint responds correctly").

## Stack

- **Language:** Python
- **LLM:** OpenAI SDK
- **Vector store / embeddings:** TBD in #B1 (Chroma or FAISS + OpenAI embeddings)
- **Web search:** Tavily
- **Observability:** Langfuse

## Planned structure

```
agent/
  llm.py            # OpenAI client wrapper (+ mock mode)
  harness.py        # inner loop (tools) + outer loop (conversation)
  state.py          # TaskState shared-state schema
  config.py         # config loading + policy engine
  memory.py         # persistent per-project memory
  context.py        # summarization, loop detection
  observability.py  # tracer (Langfuse)
  tools/            # Tool interface + base tools
  agents/           # orchestrator + 5 subagents
  rag/              # ingest + retrieve
config/agent.config.yaml
docs/               # ingested FastAPI corpus
tests/
```

## Working on this

The full work breakdown — issues, dependencies, and who owns what — lives in [`BACKLOG.md`](./BACKLOG.md) and the [GitHub issues](../../issues).

**Start here:** issue **#0 (foundation + contracts)** unblocks all parallel work — it must be merged first.

### Dev setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # core + pytest; add ".[rag]" / ".[web]" / ".[obs]" per lane
AGENT_LLM_MOCK=1 pytest -q   # smoke tests, no API key needed
```

Optional dependency groups keep core installs light: `rag` (Chroma + tiktoken),
`web` (Tavily), `obs` (Langfuse). Install only what your lane needs.

## Team

Three developers. See the ownership table in `BACKLOG.md` for lane assignments.
