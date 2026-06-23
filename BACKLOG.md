# TP Final — FastAPI Coding Agent · Backlog

Multi-agent coding agent specialized in **FastAPI**, built on top of the in-class harness, no orchestration frameworks.
Stack: **Python + OpenAI SDK**. Repo host: **GitHub**.

## How the parallelism works

The whole plan hinges on **one foundation issue (#0)** that defines the *contracts* (interfaces + shared state schema) as code stubs. Once #0 is merged, the three devs work in three independent lanes against those interfaces — coding against stubs where a real dependency from another lane hasn't landed yet. Integration happens at the end.

```
              ┌─────────────────────────┐
              │  #0 Foundation/contracts │  (blocks everything — do first, ~together)
              └────────────┬────────────┘
        ┌──────────────────┼──────────────────┐
   Lane A (Dev1)      Lane B (Dev2)       Lane C (Dev3)
   Harness/safety     RAG/Research/Obs    Subagents/cognition
        └──────────────────┼──────────────────┘
              ┌─────────────────────────┐
              │  Integration + demos +  │  (shared, end)
              │  deliverables + slides  │
              └─────────────────────────┘
```

**Ownership** (lanes below are *thematic groupings*; ownership is per this table — it crosses lanes on purpose to keep effort even):

| Dev | Theme | Issues |
|-----|-------|--------|
| **Dev 1 — Sofía** | Foundation + RAG + orchestration (+ owns final integration #I1) | #0, #B1, #B2, #C1 |
| **Dev 2** | Harness engine + instrumentation + full action chain (implement→test→review) | #A1, #A2, #B4, #C3, #C4, #C5 |
| **Dev 3** | Control/oversight layer + knowledge/cognition subagents | #A3, #A4, #B3, #C2, #C6, #C7 |

Cross-lane handoffs (all via #0 interfaces, so they don't block — mock until the real dep lands):
- #B3 Researcher (Dev 3) → consumes #B2 retrieval (Dev 1).
- #C3/#C4 (Dev 2) → use the base tools Dev 2 also owns (no cross-coupling).
- #B4 observability (Dev 2) → instruments `llm.py`/harness/tools, which Dev 2 also owns (natural fit).
- #A3 policy (Dev 3) → called by the harness (Dev 2) as `policy.check(tool_call)`; engine is standalone.

---

## Proposed package layout (defined in #0)

```
fastapi-coding-agent/
  agent/
    llm.py            # OpenAI client wrapper (single place that calls the LLM) + mock mode
    harness.py        # inner loop (tools) + outer loop (conversation)
    state.py          # TaskState shared-state schema
    config.py         # config loading + policy engine
    memory.py         # persistent per-project memory
    context.py        # summarization, loop detection, ask-for-help
    observability.py  # tracer interface (no-op until Langfuse wired)
    tools/
      __init__.py     # Tool interface + registry
      base.py         # read_file, write_file, run_command, list_files, web_search
    agents/
      __init__.py     # Subagent base interface
      orchestrator.py
      explorer.py  researcher.py  implementer.py  tester.py  reviewer.py
    rag/
      ingest.py       # fetch FastAPI docs, chunk, embed, store
      retrieve.py     # query + source attribution
  config/agent.config.yaml
  docs/               # ingested corpus (or gitignored + fetched by script)
  tests/
  README.md  pyproject.toml  .env.example
```

---

# Milestone 0 — Foundation (blocks all)

### #0 · Project scaffold + core contracts
**Owner:** Dev 1 · **Depends on:** — · **Labels:** `foundation`, `blocking`
Set up the repo and define the interfaces everyone codes against. Best done as a short pairing session so the whole team agrees on the contracts.
- Repo structure (see layout above), `pyproject.toml`/`requirements.txt`, `.env.example`, `.gitignore`, README skeleton.
- Define as documented, typed **stubs** (no full impl yet):
  - `Tool` protocol: `name`, `description`, `parameters` (JSON schema), `execute(args) -> str`, `permission` category (`read` | `write` | `command`).
  - `tools/__init__.py` registry: register + lookup tools by name.
  - `TaskState` (dataclass/pydantic): original request, plan, progress, per-subagent results, sources consulted, files modified, observations.
  - `Subagent` base: `name`, `allowed_tools`, `run(state, task) -> result`.
  - `llm.py`: single `complete(messages, tools=...) -> response` + a **mock/fake mode** so others can test without API calls.
  - `config.py`: loads `agent.config.yaml` into a typed object (stub).
  - `observability.py`: no-op tracer (`start_span`, `log`) so others instrument code before Langfuse lands.
**Acceptance:** `pip install -e .` works; all imports resolve; a trivial smoke test passes; interfaces have docstrings + type hints.

---

# Milestone 1 — Parallel build

## Lane A — Harness, tools & safety  *(#A1/#A2 → Dev 2; #A3/#A4 → Dev 3)*

### #A1 · Port & refactor the harness loop
**Depends on:** #0 · **Labels:** `harness`, `lane-a`
Refactor the in-class notebook harness into `harness.py`: inner loop (run tools until the LLM finishes its turn) + outer loop (interactive conversation, history kept between turns). Uses `llm.py` + the tool registry.
> *This is where the pinned "lift vs. merge" decision gets resolved — pick which existing source to port in here.*
**Acceptance:** can run a base tool end-to-end in a REPL; conversation persists across turns; max-iteration cap to prevent infinite loops.

### #A2 · Implement the 5 base tools
**Depends on:** #0 · **Labels:** `tools`, `lane-a`
Implement `read_file`, `write_file`, `run_command`, `list_files`, `web_search` (Tavily) against the `Tool` interface; each self-registers with the registry and declares its permission category.
**Acceptance:** unit test per tool; tools discoverable via the registry.

### #A3 · Config + policy engine (guardrails)
**Depends on:** #A1, #A2 · **Labels:** `safety`, `lane-a`
Parse `agent.config.yaml` (`read.deny`, `write.deny`, `commands.deny`, `commands.require_approval`, `workspace`); validate **every** tool call before execution. Wire into the harness.
> Improvement from TP1 reflection: don't rely on pure string matching for `rm -rf` — also constrain writes/deletes to the `workspace` dir so guardrails can't be trivially bypassed (e.g. via `shutil.rmtree`).
**Acceptance:** denied command/path is blocked; glob matching tested; `require_approval` triggers a prompt.

### #A4 · Plan mode + supervision mode
**Depends on:** #A1 · **Labels:** `harness`, `lane-a`
Carry over both modes (toggleable). Plan mode: build a step plan, let user approve/modify/reject before executing. Supervision: confirm before `write_file`/`run_command`; read-only tools run freely.
**Acceptance:** manual run shows plan approval flow + supervision prompts; modes toggle independently.

## Lane B — RAG, Researcher & observability  *(#B1/#B2 → Dev 1; #B3 → Dev 3; #B4 → Dev 2)*

### #B1 · RAG ingestion pipeline
**Depends on:** #0 · **Labels:** `rag`, `lane-b`
Fetch FastAPI's official docs (markdown from `tiangolo/fastapi/docs`), chunk them, embed (OpenAI embeddings), store in a vector DB (Chroma or FAISS). Provide a `python -m agent.rag.ingest` entrypoint.
**Acceptance:** running ingest populates the store; chunking strategy + chosen store documented (feeds deliverable #E3).

### #B2 · RAG retrieval + source attribution
**Depends on:** #B1 · **Labels:** `rag`, `lane-b`
`retrieve(query, k)` returns top-k chunks **with provenance** (source file/section + score). Expose as a tool or internal API the Researcher calls.
**Acceptance:** a sample query returns relevant FastAPI chunks with cited sources.

### #B3 · Researcher subagent
**Depends on:** #B2, #0 (Subagent iface) · #A2 web_search (stub OK) · **Labels:** `subagent`, `lane-b`
RAG-first: query the vector store; if evidence is insufficient, fall back to web search (prioritize official docs). Records sources to shared state and **labels each piece of info by origin** (RAG / web / inference).
**Acceptance:** returns an answer plus the cited sources and their origin labels.

### #B4 · Observability integration (Langfuse)
**Depends on:** #0 (tracer iface) · **Labels:** `observability`, `lane-b`
Implement the tracer against the no-op interface. Log: prompts, model, LLM calls, tools invoked, retrieved docs, web searches, iterations, errors, latency, tokens, estimated cost, final result.
> Parallel-safe: the interface is fixed in #0, so wrapping `llm.py`/tools only requires the agreed methods, not others' internals.
**Acceptance:** one full run produces a complete trace in Langfuse (screenshot for #E2/#E3).

## Lane C — Subagents & cognition  *(#C3/#C4/#C5 → Dev 2; #C2/#C6/#C7 → Dev 3)*

### #C2 · Explorer subagent
**Depends on:** #0, #A2 (stub OK) · **Labels:** `subagent`, `lane-c`
Understands the target repo: structure, architecture, dependencies, conventions, relevant files. Writes findings to shared state. Read-only tools only.
**Acceptance:** given a FastAPI repo, produces a structured summary in `TaskState`.

### #C3 · Implementer subagent
**Depends on:** #0, #A2 (stub OK) · **Labels:** `subagent`, `lane-c`
Proposes/applies code changes based on Explorer + Researcher findings. Has write access.
**Acceptance:** given findings + a task, edits files and records the diff/changed files in state.

### #C4 · Tester subagent
**Depends on:** #0, #A2 (stub OK) · **Labels:** `subagent`, `lane-c`
Validates results via group-defined checks: run `pytest` / start server & hit endpoint / lint / build. Reports pass/fail + output.
**Acceptance:** runs checks and writes structured results to state.

### #C5 · Reviewer subagent
**Depends on:** #0 · **Labels:** `subagent`, `lane-c`
Reviews the diff/changes against the original request; approves or sends back with reasons.
**Acceptance:** given a diff + the request, returns an approve/reject verdict with rationale.

### #C6 · Persistent project memory
**Depends on:** #0 · **Labels:** `memory`, `lane-c`
Per-project memory beyond conversation history: detected architecture, important files, dependencies, useful commands, conventions, decisions, investigated bugs, session summaries. Persist to disk (JSON/files); load on startup.
**Acceptance:** memory survives restart; a second run reuses prior findings.

### #C7 · Context management & loop detection
**Depends on:** #0, #A1 · **Labels:** `cognition`, `lane-c`
Summarize old history; keep key decisions; avoid sending the whole repo/history each turn. Detect no-progress loops (same command → same error, re-reading files without new info) and change strategy / stop / ask for help. Recognize insufficient-evidence situations (ambiguous request, missing docs, permission blocks) and explain what's missing.
**Acceptance:** a deliberately looping task is caught and the agent changes strategy or asks for help.

## Glue (Dev 1 — Sofía, after subagents have stubs)

### #C1 · Orchestrator / main agent
**Depends on:** #0; integrates with all subagents · **Labels:** `orchestrator`, `lane-a`
Receives the task, owns shared `TaskState`, coordinates subagents (explore → research → implement → test → review), can call tools directly. Drives stub subagents first, real ones as they land.
**Acceptance:** runs the full sequence end-to-end with stubbed subagents, then with real ones.

---

# Milestone 2 — Integration & deliverables (shared)

### #E1 · Use-case setup *(can start early — no code deps)*
**Labels:** `usecase`, `docs`
Choose/clone the target FastAPI repo, write its `agent.config.yaml`, and define the concrete goal + success criterion (e.g. "add a `POST /users` endpoint with email validation; success = `pytest` passes and endpoint returns 201").

### #I1 · End-to-end integration
**Depends on:** all Milestone 1 · **Labels:** `integration`
Wire all subagents into the orchestrator; run the full flow on the target repo; fix integration gaps.

### #E2 · Two demo runs + evidence capture
**Depends on:** #I1, #B4 · **Labels:** `demo`
Run ≥2 tasks on the use case (one must use RAG + show retrieved sources; one must use project memory; ideally one where the agent changes strategy / asks for help). Capture output, retrieved sources, and Langfuse trace screenshots.

### #E3 · Deliverable docs
**Depends on:** #I1 · **Labels:** `docs`
README (install/config/run), architecture writeup (orchestrator + each subagent + shared-state structure), RAG documentation (sources, chunking, embeddings, store), reflection (what worked, what failed, loops/insufficient-evidence detection, improvements).

### #E4 · Exam presentation / slides
**Depends on:** #E2, #E3 · **Labels:** `presentation`
Slides for the final: problem, architecture diagram, live/recorded demo, observability trace, learnings.

---

## Dependency cheat-sheet
- **#0 blocks everything.** Merge it first.
- Lanes A/B/C run in parallel against #0's interfaces (use stubs for cross-lane deps).
- `#A3 → #A1,#A2` · `#B2 → #B1` · `#B3 → #B2` · `#C7 → #A1` · `#C1` integrates subagents · `#I1 → all` · `#E2 → #I1,#B4` · `#E4 → #E2,#E3`.
- **#E1 has no code deps** — start it on day one.
