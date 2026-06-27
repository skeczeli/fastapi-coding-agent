"""Orchestrator / main agent (#C1, Dev 1).

The main agent. Receives the task, owns the shared ``TaskState``, and coordinates
the subagents via the *agents-as-tools* pattern: each subagent (Explorer,
Researcher, Implementer, Tester, Reviewer) is exposed to the orchestrator's LLM
as a callable tool. The LLM decides the real order — the intended flow is
explore → research → implement → test → review, but it can revisit or skip steps.
All subagents read/write the same ``TaskState``.

The orchestrator reuses the shared ``run_loop`` primitive (the same one every
subagent runs), handing it the subagent-as-tool adapters as its toolset. Because
``run_loop`` dispatches over the tools it's given (not the global registry), the
adapters are built per run and passed in directly — no global registration.

It wraps the run with the cross-cutting concerns it owns:
- **Project memory (#C6)**: load persisted knowledge before delegating and save
  it after, so findings outlive a single conversation.
- **Context management / loop detection (#C7)** lives inside ``run_loop`` itself.

Drives stub subagents today; the real subagents (#C2–#C5, #B3) and the real
memory/context impls drop in unchanged since they all satisfy the same contracts.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent import harness
from agent.agents import Subagent
from agent.agents.explorer import Explorer
from agent.agents.implementer import Implementer
from agent.agents.researcher import Researcher
from agent.agents.reviewer import Reviewer
from agent.agents.tester import Tester
from agent.memory import ProjectMemory
from agent.state import TaskState
from agent.tools import Permission

ORCHESTRATOR_PROMPT = """You are the orchestrator of a multi-agent coding agent \
specialized in FastAPI. You own the task and coordinate specialist subagents, each \
available to you as a tool. You delegate by calling the right subagent with a \
clear, specific task — you don't write code or run commands yourself.

Your subagents:
- explorer: understands the target repo (structure, deps, conventions, relevant files).
- researcher: answers FastAPI/library questions. RAG-first over the official docs, \
web search only as a fallback; reports which sources it used and labels each \
finding by origin (rag / web / inference).
- implementer: applies the code changes based on the explorer's and researcher's findings.
- tester: runs checks (pytest, lint, start server & hit an endpoint) and reports pass/fail.
- reviewer: reviews the changes against the original request and approves or sends back.

Project memory:
You are given the project's persistent memory (detected architecture, key files, \
conventions, decisions, prior session summaries). Consult it BEFORE delegating — \
don't re-explore what's already known. As the task progresses, note durable \
findings worth keeping across runs: architecture you confirmed, useful commands, \
conventions, and decisions you made (with the reason). Memory outlives this \
conversation; the working state does not.

Knowing when NOT to push forward is part of your job. Stop and ask the user for \
help — explaining what you tried, what's missing, and what you need to continue — \
when:
- the request is ambiguous or underspecified;
- the evidence is insufficient (no RAG hits, no docs, an undiagnosed error);
- a policy blocks a required action, or a change looks too risky;
- you're looping without progress (same command → same error, re-reading files \
without learning anything new): change strategy, replan, or stop — don't retry the \
same approach.

Typical flow: explore → research → implement → test → review. Adapt as needed — \
re-research if the implementer is blocked, re-test after fixes. Each subagent \
writes its findings to shared state, so pass each one the context it needs \
(a short summary of prior findings, not the whole history).

When the task is complete and the reviewer approves, stop and give a short summary \
of what was done and which sources (and their origin: repo / memory / rag / web / \
inference) informed it."""


@dataclass
class SubagentTool:
    """Adapts a ``Subagent`` into a ``Tool`` (the agents-as-tools pattern).

    The orchestrator's LLM calls this like any other tool; ``execute`` forwards
    the requested task to the wrapped subagent, which runs its own harness loop
    and mutates the shared ``state``. Bound to a single run's ``state`` at
    construction, so it's transient — never registered in the global registry.

    Permission is ``command``: delegating to a subagent can transitively read,
    write, or run commands, so #A3's policy engine should gate it conservatively.
    The real per-tool enforcement still happens inside the subagent's own loop,
    against its ``allowed_tools`` subset.
    """

    subagent: Subagent
    state: TaskState
    permission: Permission = "command"

    @property
    def name(self) -> str:
        return self.subagent.name

    @property
    def description(self) -> str:
        return f"Delegate a task to the {self.subagent.name} subagent."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        f"A clear, self-contained task for the {self.subagent.name} "
                        "subagent, including the context it needs."
                    ),
                }
            },
            "required": ["task"],
        }

    def execute(self, args: dict) -> str:
        task = args.get("task", "")
        self.state.progress.append(f"→ {self.subagent.name}: {task}")
        return self.subagent.run(self.state, task)


@dataclass
class RememberProjectTool:
    """Tool that lets the orchestrator persist findings to project memory.

    Built per-run with the current ``ProjectMemory`` instance — same pattern
    as ``SubagentTool``. Not globally registered.
    """

    memory: ProjectMemory
    name: str = "remember_project"
    description: str = (
        "Persist a finding to project memory so it's available in future runs. "
        "Use this whenever you learn something durable about the project: "
        "architecture, key files, dependencies, useful commands, conventions, "
        "decisions, or bugs investigated."
    )
    permission: Permission = "write"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": [
                        "architecture",
                        "important_files",
                        "dependencies",
                        "commands",
                        "conventions",
                        "decisions",
                        "bugs",
                    ],
                    "description": "The memory category to store the finding under.",
                },
                "content": {
                    "type": "string",
                    "description": "The finding to persist.",
                },
            },
            "required": ["category", "content"],
        }

    def execute(self, args: dict) -> str:
        category = args.get("category", "")
        content = args.get("content", "")
        if not category or not content:
            return "[error] both 'category' and 'content' are required"
        if category == "architecture":
            self.memory.set_architecture(content)
        else:
            self.memory.remember(category, content)
        return f"Remembered in '{category}': {content}"


def default_subagents() -> list[Subagent]:
    """The standard roster, in the intended explore→…→review order.

    Returns fresh instances each call so two runs never share subagent state.
    """
    return [Explorer(), Researcher(), Implementer(), Tester(), Reviewer()]


def _memory_preamble(mem: ProjectMemory) -> str:
    """Render persisted memory as a context block appended to the first message."""
    if not mem.data:
        return ""
    lines = "\n".join(f"- {key}: {value}" for key, value in mem.data.items())
    return f"\n\n## Project memory (from prior runs)\n{lines}"


def run(
    request: str,
    subagents: list[Subagent] | None = None,
    max_iters: int = 25,
    memory_path: str = ".agent_memory",
) -> TaskState:
    """Run a full coding task end-to-end and return the final shared state.

    Loads project memory, builds the shared ``TaskState``, wraps each subagent
    as a tool, and lets the orchestrator LLM drive them through ``run_loop``.
    The final summary is recorded on the state, memory is persisted, and the
    state is returned for inspection (sources consulted, files modified,
    per-subagent results, progress log).

    ``require_approval`` commands (run by the Tester/Implementer subagents) are
    gated by the process-wide approval handler — set ``harness.set_approval_fn``
    (the CLI wires a stdin prompt) to get confirm-before-run; otherwise they are
    denied.

    Args:
        request: The user's coding task, verbatim.
        subagents: Roster to coordinate. Defaults to the standard five.
        max_iters: Hard cap on orchestration steps (passed to ``run_loop``).
        memory_path: Directory where per-project memory is persisted (default: .agent_memory/).

    Returns:
        The final ``TaskState`` after the run completes or hits the cap.
    """
    state = TaskState(request=request)
    mem = ProjectMemory.load(memory_path)

    # Record loaded memory as consulted sources (origin="memory"), so attribution
    # reflects what prior-run knowledge was injected into the agent's context —
    # otherwise a memory-only answer looks like it had no evidence at all.
    for key, value in mem.data.items():
        state.add_source("memory", key, snippet=str(value)[:200])

    roster = subagents if subagents is not None else default_subagents()
    tool_list: list = [SubagentTool(subagent=sa, state=state) for sa in roster]
    tool_list.append(RememberProjectTool(memory=mem))

    summary = harness.run_loop(
        system_prompt=ORCHESTRATOR_PROMPT,
        tool_list=tool_list,
        state=state,
        user_msg=request + _memory_preamble(mem),
        max_iters=max_iters,
    )

    state.subagent_results["orchestrator"] = summary
    # ``run_loop`` returns a "[harness] stopped: ..." sentinel when it hits the
    # iteration cap or a no-progress loop — that's the loop bailing out, not the
    # orchestrator's answer. Flag it so callers (and #C7's ask-for-help path)
    # can replan or surface it instead of treating it as a clean result.
    if summary.startswith("[harness] stopped:"):
        state.note(f"orchestration halted before completion: {summary}")
    else:
        state.progress.append(f"orchestrator done: {summary}")
    mem.save(memory_path)
    return state


# --------------------------------------------------------------------------- #
# CLI entrypoint (#I1) — `python -m agent.agents.orchestrator "<task>"`
# The multi-agent counterpart to `python -m agent` (single agent).
# --------------------------------------------------------------------------- #


def _section(title: str, body: list[str]) -> list[str]:
    """A titled block for the report, or a '(none)' placeholder if empty."""
    if not body:
        return [f"\n{title}: (none)"]
    return [f"\n{title}:", *body]


def _dedup_sources(sources: list) -> list[str]:
    """Render sources, collapsing duplicates by (origin, ref).

    The Researcher reruns ``rag_search`` with rephrased queries, so the same chunk
    is recorded many times. Keep each distinct source once (with its best score),
    in first-seen order, so the list stays readable instead of a wall of repeats.
    """
    best: dict[tuple, float | None] = {}
    order: list[tuple] = []
    for s in sources:
        key = (s.origin, s.ref)
        if key not in best:
            order.append(key)
            best[key] = s.score
        elif s.score is not None and (best[key] is None or s.score > best[key]):
            best[key] = s.score
    out: list[str] = []
    for origin, ref in order:
        score = best[(origin, ref)]
        out.append(f"  - [{origin}] {ref}" + (f"  (score={score:.2f})" if score is not None else ""))
    return out


def render_state(state: TaskState) -> str:
    """Render the final ``TaskState`` as a human-readable report.

    Surfaces what the assignment asks a run to show: the orchestrator's answer,
    the sources consulted *with their origin labels* (rag / web / repo / memory /
    inference), the files touched, what each subagent reported, and the
    delegation/progress log. Pure (returns a string) so it's easy to test.
    """
    lines = ["=" * 70, "ORCHESTRATOR RESULT", "=" * 70, f"\nRequest: {state.request}"]

    summary = state.subagent_results.get("orchestrator", "")
    if summary:
        lines.append(f"\nFinal answer:\n{summary}")

    lines += _section("Sources consulted", _dedup_sources(state.sources))

    files = [f"  - {f}" for f in state.files_modified]
    lines += _section("Files modified", files)

    results = [
        f"  - {name}: {_truncate(str(result))}"
        for name, result in state.subagent_results.items()
        if name != "orchestrator"
    ]
    lines += _section("Per-subagent results", results)

    lines += _section("Progress log", [f"  {p}" for p in state.progress])
    lines += _section("Observations", [f"  - {o}" for o in state.observations])

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def _truncate(text: str, limit: int = 300) -> str:
    """Clip long subagent results so the report stays scannable."""
    return text if len(text) <= limit else text[:limit] + "…"


def _build_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m agent.agents.orchestrator",
        description="Run the multi-agent FastAPI coding agent on a task.",
    )
    parser.add_argument("task", nargs="+", help="The coding task for the agent to perform.")
    parser.add_argument(
        "--max-iters", type=int, default=25, help="Orchestration step cap (default: 25)."
    )
    parser.add_argument(
        "--memory-path",
        default=".agent_memory",
        help="Where per-project memory is persisted (default: .agent_memory/).",
    )
    return parser


def _report_api_error(err: Exception) -> int:
    """Turn a crashed run into a readable message instead of a traceback.

    Recognizes the common OpenAI failure modes (no credit, bad key, rate limit)
    and prints what to do about each; anything else is shown verbatim. Always
    returns exit code ``1`` so scripts can detect the failure.
    """
    text = str(err)
    if "insufficient_quota" in text or "exceeded your current quota" in text:
        hint = (
            "OpenAI account has no credit (insufficient_quota). Add billing at "
            "https://platform.openai.com/account/billing or use a funded API key."
        )
    elif "invalid_api_key" in text or "Incorrect API key" in text:
        hint = "OpenAI rejected the API key. Check OPENAI_API_KEY in your .env."
    elif "rate_limit" in text.lower():
        hint = "Hit the OpenAI rate limit. Wait a moment and retry."
    else:
        hint = f"Unexpected error: {type(err).__name__}: {text}"
    print(f"\n[run failed] {hint}")
    return 1


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m agent.agents.orchestrator "<task>"``.

    Loads ``.env``, installs the Langfuse tracer if configured, runs the
    orchestrator over the five real subagents, prints the final ``TaskState``
    report, and flushes traces before exit. Returns ``0`` on a clean finish,
    ``2`` if the run halted on a loop / iteration cap, ``1`` on a config error.
    """
    import os

    # Load .env so OPENAI / TAVILY / LANGFUSE keys are present before any call.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    from agent import observability, policy

    args = _build_parser().parse_args(argv)
    request = " ".join(args.task)

    if not os.getenv("AGENT_LLM_MOCK") and not os.getenv("OPENAI_API_KEY"):
        print(
            "No OPENAI_API_KEY found. Add it to .env (see .env.example), or set "
            "AGENT_LLM_MOCK=1 to run the loop offline with canned responses."
        )
        return 1

    tracer = observability.init_tracer()
    tracing_on = not isinstance(tracer, observability.NoopTracer)
    print(f"Observability: {'Langfuse trace on' if tracing_on else 'off (no LANGFUSE keys)'}")
    print(f"Task: {request}\n")

    # Confirm-before-run for require_approval commands (pip install, git commit, …).
    # Set process-wide so it reaches the subagents' loops (where commands run).
    def approval_fn(description: str) -> bool:
        print(f"[approval required] {description}")
        try:
            return input("Approve? [y/n]: ").strip().lower() in ("y", "yes", "s", "si")
        except (EOFError, KeyboardInterrupt):
            return False

    policy.set_approval_fn(approval_fn)

    try:
        state = run(request, max_iters=args.max_iters, memory_path=args.memory_path)
    except Exception as err:  # noqa: BLE001 — CLI boundary: report, don't crash.
        observability.flush()
        return _report_api_error(err)
    finally:
        # Flush buffered traces so the run shows up in Langfuse before exit.
        observability.flush()

    print(render_state(state))

    halted = str(state.subagent_results.get("orchestrator", "")).startswith("[harness] stopped:")
    return 2 if halted else 0


if __name__ == "__main__":
    raise SystemExit(main())
