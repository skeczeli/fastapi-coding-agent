"""Smoke test for #0 — proves the scaffold imports and the contracts wire up.

Must pass with no API key (uses AGENT_LLM_MOCK). This is the acceptance check
for #0: all imports resolve, the registry works, TaskState builds, and the LLM
mock path returns a response.
"""

import os

os.environ["AGENT_LLM_MOCK"] = "1"  # no real API calls in tests

from dataclasses import dataclass

from agent import config, harness, llm, tools
from agent.agents import Subagent
from agent.agents.orchestrator import run as orchestrate
from agent.observability import NoopTracer, get_tracer
from agent.rag.retrieve import retrieve
from agent.state import Source, TaskState
from agent.tools import Tool


def test_all_modules_import():
    # Importing the packages above already exercises the tree; assert versions.
    import agent

    assert agent.__version__


def test_base_tools_registered():
    names = {t.name for t in tools.all_tools()}
    assert {"read_file", "write_file", "run_command", "list_files", "web_search"} <= names
    assert tools.get("read_file").permission == "read"


def test_tool_schemas_openai_shape():
    schema = tools.schemas([tools.get("read_file")])[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "read_file"


def test_taskstate_and_source_attribution():
    state = TaskState(request="add POST /users")
    src = state.add_source("rag", "docs/tutorial/body.md", "use a Pydantic model")
    assert isinstance(src, Source)
    assert state.sources[0].origin == "rag"


def test_custom_tool_registers_and_runs():
    @dataclass
    class Echo:
        name: str = "echo"
        description: str = "echo back"
        parameters: dict = None
        permission: str = "read"

        def execute(self, args: dict) -> str:
            return args.get("x", "")

    echo = Echo(parameters={"type": "object", "properties": {}})
    assert isinstance(echo, Tool)  # satisfies the Protocol
    tools.register(echo)
    assert tools.get("echo").execute({"x": "hi"}) == "hi"


def test_llm_complete_mock():
    resp = llm.complete([{"role": "user", "content": "ping"}])
    assert isinstance(resp, llm.LLMResponse)
    assert "ping" in resp.content


def test_run_loop_returns_with_mock():
    state = TaskState(request="noop")
    # Mock LLM returns no tool calls, so the loop returns immediately.
    out = harness.run_loop("you are a test agent", [], state, "do nothing")
    assert isinstance(out, str)


def test_retrieve_contract():
    assert retrieve("query", k=3) == []  # stub until #B2


def test_config_loads_example():
    cfg = config.load_config("config/agent.config.yaml")
    assert cfg.workspace == "./workspace"
    assert "rm -rf*" in cfg.commands.deny


def test_orchestrator_stub_returns_state():
    state = orchestrate("add an endpoint")
    assert isinstance(state, TaskState)
    assert state.request == "add an endpoint"


def test_noop_tracer():
    tracer = get_tracer()
    assert isinstance(tracer, NoopTracer)
    with tracer.span("x"):
        tracer.log(event="ok")
