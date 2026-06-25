"""``python -m agent`` — interactive REPL for the single-agent coding harness (#A1).

The ported in-class TP agent: one agent, the five base tools, a persistent chat.
The multi-agent orchestrator is a separate entrypoint (``agent.agents.orchestrator``).

Run it::

    python -m agent          # needs OPENAI_API_KEY (in .env or the environment)

Set ``AGENT_LLM_MOCK=1`` to try the loop offline with canned LLM responses.
"""

from __future__ import annotations

import os

from agent import harness, tools
from agent.modes import HarnessMode
from agent.state import TaskState


def main() -> None:
    # Load .env so OPENAI_API_KEY (and TAVILY_API_KEY for web_search) are present
    # before the first LLM/tool call. No-op if python-dotenv isn't installed.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if not os.getenv("AGENT_LLM_MOCK") and not os.getenv("OPENAI_API_KEY"):
        print(
            "No OPENAI_API_KEY found. Add it to .env (see .env.example), or set "
            "AGENT_LLM_MOCK=1 to run the loop offline with canned responses."
        )
        return

    state = TaskState(request="(interactive session)")
    # all_tools() returns the base tools, which self-register on import.
    tool_list = tools.all_tools()
    mode = HarnessMode()

    print("FastAPI coding agent — interactive harness. Type 'exit' to quit.")
    print(f"Tools available: {', '.join(t.name for t in tool_list)}")
    print("Toggle modes: /plan, /supervision\n")
    harness.converse(tool_list, state, mode=mode)


if __name__ == "__main__":
    main()
