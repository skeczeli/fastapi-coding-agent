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
    memory_path: str = ".agent_memory.json",
) -> TaskState:
    """Run a full coding task end-to-end and return the final shared state.

    Loads project memory, builds the shared ``TaskState``, wraps each subagent
    as a tool, and lets the orchestrator LLM drive them through ``run_loop``.
    The final summary is recorded on the state, memory is persisted, and the
    state is returned for inspection (sources consulted, files modified,
    per-subagent results, progress log).

    Args:
        request: The user's coding task, verbatim.
        subagents: Roster to coordinate. Defaults to the standard five.
        max_iters: Hard cap on orchestration steps (passed to ``run_loop``).
        memory_path: Where per-project memory is persisted (no-op until #C6).

    Returns:
        The final ``TaskState`` after the run completes or hits the cap.
    """
    state = TaskState(request=request)
    mem = ProjectMemory.load(memory_path)

    roster = subagents if subagents is not None else default_subagents()
    tool_list = [SubagentTool(subagent=sa, state=state) for sa in roster]

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
