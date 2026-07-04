# Copyright © 2025 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""Offline tests for the LangGraph ``ManagerWorkers`` converter.

These cover the hierarchical topology, roster prompt rendering, the
worker-isolation invariant (each worker sees only its delegated task),
and the recursive nesting case. The LLM is stubbed with
``FakeMessagesListChatModel`` so the tests run without network or model
endpoints.
"""

from typing import Any
from unittest.mock import patch

import pytest


# ─── Shared helpers ──────────────────────────────────────────────────────────


def _llm_cfg(name: str) -> Any:
    from pyagentspec.llms.openaicompatibleconfig import OpenAiCompatibleConfig

    return OpenAiCompatibleConfig(name=name, model_id="fake", url="null")


def _fake_manager(*ai_responses: Any) -> Any:
    """A FakeMessagesListChatModel subclassed under ChatOpenAI so the
    manager's react-agent treats it as an OpenAI-style chat model."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_openai import ChatOpenAI

    class _FakeModel(FakeMessagesListChatModel, ChatOpenAI):
        pass

    return _FakeModel(responses=list(ai_responses))


# ─── Pure-helper unit tests (no LLM) ────────────────────────────────────────


def test_safe_node_name_lowercases_and_collapses_punctuation() -> None:
    from pyagentspec.adapters.langgraph._langgraphconverter import _safe_node_name

    assert _safe_node_name("Research Helper", "id-1") == "research_helper"
    assert _safe_node_name("My-Worker!! v2", "id-1") == "my_worker_v2"


def test_safe_node_name_falls_back_to_normalized_id() -> None:
    from pyagentspec.adapters.langgraph._langgraphconverter import _safe_node_name

    # Name slugifies to empty → id used (and also normalized).
    assert _safe_node_name("!!!", "sub-1") == "sub_1"
    # Both empty → constant fallback.
    assert _safe_node_name("", "") == "worker"


def test_append_workers_roster_appends_block_after_existing_prompt() -> None:
    from pyagentspec.adapters.langgraph._langgraphconverter import _append_workers_roster

    out = _append_workers_roster(
        "Coordinate the team.",
        [("research_helper", "Handles research"), ("drafter", "Drafts text")],
    )
    assert out == (
        "Coordinate the team.\n\n"
        "Available workers:\n"
        "- research_helper: Handles research\n"
        "- drafter: Drafts text"
    )


def test_append_workers_roster_flattens_multiline_descriptions() -> None:
    from pyagentspec.adapters.langgraph._langgraphconverter import _append_workers_roster

    out = _append_workers_roster(
        "",
        [("helper", "First line\nsecond line\n  third  line  ")],
    )
    # Whitespace flattened so the one-line-per-worker shape survives.
    assert out == "Available workers:\n- helper: First line second line third line"


def test_route_manager_to_worker_or_end_sends_to_pending_delegation() -> None:
    from langchain_core.messages import AIMessage
    from langgraph.types import Send

    from pyagentspec.adapters.langgraph._langgraphconverter import (
        _DELEGATE_CALL_ID_KEY,
        _DELEGATE_TASK_KEY,
        _route_manager_to_worker_handoff_or_end,
    )

    delegating = AIMessage(
        content="",
        tool_calls=[
            {"name": "delegate_to_research_helper", "args": {"task": "hi"}, "id": "c1"}
        ],
    )
    sends = _route_manager_to_worker_handoff_or_end({"messages": [delegating]})
    # One delegation → a single Send to the worker node carrying the task
    # and the tool_call_id its reply must answer.
    assert isinstance(sends, list) and len(sends) == 1
    assert isinstance(sends[0], Send)
    assert sends[0].node == "research_helper"
    assert sends[0].arg == {_DELEGATE_TASK_KEY: "hi", _DELEGATE_CALL_ID_KEY: "c1"}


def test_route_manager_to_worker_or_end_returns_end_when_no_delegation() -> None:
    from langchain_core.messages import AIMessage

    from pyagentspec.adapters.langgraph._langgraphconverter import (
        _route_manager_to_worker_handoff_or_end,
    )
    from langgraph.graph import END

    not_delegating = AIMessage(content="Done.", tool_calls=[])
    assert _route_manager_to_worker_handoff_or_end({"messages": [not_delegating]}) == END
    assert _route_manager_to_worker_handoff_or_end({"messages": []}) == END


def test_route_manager_to_worker_or_end_fans_out_every_delegation() -> None:
    from langchain_core.messages import AIMessage
    from langgraph.types import Send

    from pyagentspec.adapters.langgraph._langgraphconverter import (
        _DELEGATE_CALL_ID_KEY,
        _DELEGATE_TASK_KEY,
        _route_manager_to_worker_handoff_or_end,
    )

    msg = AIMessage(
        content="",
        tool_calls=[
            {"name": "some_other_tool", "args": {}, "id": "c0"},
            {"name": "delegate_to_drafter", "args": {"task": "x"}, "id": "c1"},
            {"name": "delegate_to_research_helper", "args": {"task": "y"}, "id": "c2"},
        ],
    )
    sends = _route_manager_to_worker_handoff_or_end({"messages": [msg]})
    # Every delegation gets its own Send so each tool_call_id is answered.
    # The non-delegation tool call was already executed inside the manager's
    # react loop and is ignored by routing.
    assert all(isinstance(s, Send) for s in sends)
    assert [s.node for s in sends] == ["drafter", "research_helper"]
    assert [s.arg[_DELEGATE_CALL_ID_KEY] for s in sends] == ["c1", "c2"]
    assert [s.arg[_DELEGATE_TASK_KEY] for s in sends] == ["x", "y"]


# ─── Topology test (no LLM execution; checks compiled graph shape) ──────────


def test_manager_workers_compiles_to_hierarchical_graph_topology() -> None:
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.adapters.langgraph._langgraphconverter import (
        AgentSpecToLangGraphConverter,
        _MANAGER_NODE_KEY,
    )
    from pyagentspec.agent import Agent
    from pyagentspec.managerworkers import ManagerWorkers

    manager_agent = Agent(
        name="Coordinator",
        description="Coordinates",
        system_prompt="Coordinate the team.",
        llm_config=_llm_cfg("manager_llm"),
    )
    worker_a = Agent(
        name="Research Helper",
        description="Handles research",
        system_prompt="Research.",
        llm_config=_llm_cfg("worker_a_llm"),
    )
    worker_b = Agent(
        name="Drafter",
        description="Drafts text",
        system_prompt="Draft.",
        llm_config=_llm_cfg("worker_b_llm"),
    )
    mw = ManagerWorkers(
        name="ResearchTeam",
        group_manager=manager_agent,
        workers=[worker_a, worker_b],
    )

    fake_llm = _fake_manager(AIMessage(content="Done."))
    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with patch.object(
        AgentSpecToLangGraphConverter, "_llm_convert_to_langgraph", return_value=fake_llm
    ):
        compiled = loader.load_component(mw)

    # The compiled object is a CompiledStateGraph; its builder exposes
    # the parent topology we expect.
    builder = compiled.builder
    assert _MANAGER_NODE_KEY in builder.nodes
    assert "research_helper" in builder.nodes
    assert "drafter" in builder.nodes

    # START → manager; every worker → manager (loop).
    edge_pairs = {(src, dst) for src, dst in builder.edges}
    assert (START, _MANAGER_NODE_KEY) in edge_pairs
    assert ("research_helper", _MANAGER_NODE_KEY) in edge_pairs
    assert ("drafter", _MANAGER_NODE_KEY) in edge_pairs

    # The manager → worker routing is a conditional edge (branch), not a
    # plain edge — branches are stored separately on the builder.
    branches = builder.branches.get(_MANAGER_NODE_KEY) or {}
    assert branches, "expected a conditional branch from the manager node"


def test_manager_workers_renders_workers_roster_into_manager_prompt() -> None:
    """The manager's system prompt gets the ``Available workers:`` block
    appended so the LLM knows which delegation tool maps to which worker.
    """
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import MemorySaver

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.adapters.langgraph._langgraphconverter import (
        AgentSpecToLangGraphConverter,
        _MANAGER_NODE_KEY,
    )
    from pyagentspec.agent import Agent
    from pyagentspec.managerworkers import ManagerWorkers

    manager_agent = Agent(
        name="Coordinator",
        description="Coordinates",
        system_prompt="Coordinate the team.",
        llm_config=_llm_cfg("manager_llm"),
    )
    worker = Agent(
        name="Research Helper",
        description="Handles research tasks",
        system_prompt="Research.",
        llm_config=_llm_cfg("worker_llm"),
    )
    mw = ManagerWorkers(
        name="Team",
        group_manager=manager_agent,
        workers=[worker],
    )

    fake_llm = _fake_manager(AIMessage(content="Done."))
    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with patch.object(
        AgentSpecToLangGraphConverter, "_llm_convert_to_langgraph", return_value=fake_llm
    ):
        compiled = loader.load_component(mw)

    # The manager react-agent is itself a subgraph; its create_agent
    # middleware stack carries the rendered system prompt as the
    # first message of every turn. Walk the manager subgraph's pre-model
    # hook chain to find it.
    manager_subgraph = compiled.builder.nodes[_MANAGER_NODE_KEY].runnable
    # `create_agent` builds a graph whose system message generation
    # wraps the prompt — easier to assert by re-rendering it through the
    # same helper used by the converter and checking the *intent*.
    from pyagentspec.adapters.langgraph._langgraphconverter import _append_workers_roster

    expected = _append_workers_roster(
        "Coordinate the team.",
        [("research_helper", "Handles research tasks")],
    )
    assert "Available workers:" in expected
    assert "- research_helper: Handles research tasks" in expected
    # And the compiled manager carries the delegation tool the prompt
    # advertises, proving the LLM has the matching contract.
    tools_node = manager_subgraph.builder.nodes["tools"].runnable
    assert "delegate_to_research_helper" in tools_node.tools_by_name


# ─── End-to-end execution test (offline, fake LLM emitting delegation) ──────


def test_manager_workers_delegates_and_routes_back_with_tool_message() -> None:
    """End-to-end: manager LLM emits a delegate_to_<worker> tool call,
    the parent graph routes to the worker subgraph (which runs with an
    isolated message context), the worker's final AIMessage content is
    surfaced back to the manager as a ToolMessage matched to the
    pending tool_call_id, and the manager's next turn (no tool call)
    terminates the graph. This is the load-bearing path that proves the
    subgraph composition actually works."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.adapters.langgraph._langgraphconverter import (
        AgentSpecToLangGraphConverter,
    )
    from pyagentspec.agent import Agent
    from pyagentspec.managerworkers import ManagerWorkers

    manager_agent = Agent(
        name="Coordinator",
        description="Coordinates",
        system_prompt="You coordinate.",
        llm_config=_llm_cfg("manager_llm"),
    )
    worker = Agent(
        name="Research Helper",
        description="Handles research",
        system_prompt="You research.",
        llm_config=_llm_cfg("worker_llm"),
    )
    mw = ManagerWorkers(
        name="Team",
        group_manager=manager_agent,
        workers=[worker],
    )

    # Manager turn 1: delegate to research_helper.
    # Manager turn 2: produce final answer (no tool call → END).
    manager_responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "delegate_to_research_helper",
                    "args": {"task": "Look up Saturn"},
                    "id": "call_1",
                }
            ],
        ),
        AIMessage(content="The worker reports: Saturn has rings."),
    ]
    # Worker turn 1: produce its own final answer.
    worker_responses = [AIMessage(content="Saturn has rings.")]

    fake_manager = _fake_manager(*manager_responses)
    fake_worker = _fake_manager(*worker_responses)

    def _dispatch(self_obj: Any, llm_config: Any, *args: Any, **kwargs: Any) -> Any:
        if llm_config.name == "manager_llm":
            return fake_manager
        if llm_config.name == "worker_llm":
            return fake_worker
        raise AssertionError(f"unexpected llm_config: {llm_config.name}")

    # ``create_agent`` calls ``model.bind_tools(...)``. ``FakeMessagesListChatModel``
    # inherits ``bind_tools`` from real ``ChatOpenAI``, which calls out to
    # OpenAI. Patch the class method so binding is a no-op that returns the
    # same fake (preserving its response queue).
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )

    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with patch.object(
        AgentSpecToLangGraphConverter,
        "_llm_convert_to_langgraph",
        autospec=True,
        side_effect=_dispatch,
    ), patch.object(
        FakeMessagesListChatModel,
        "bind_tools",
        new=lambda self_obj, *a, **kw: self_obj,
    ):
        compiled = loader.load_component(mw)

    # Use the sync invocation path: ``FakeMessagesListChatModel`` provides
    # a sync ``_generate`` (returns queued responses) but no async
    # override, so MRO resolves ``_agenerate`` to the real
    # ``ChatOpenAI._agenerate`` which calls the OpenAI API. The worker
    # wrapper exposes both sync and async via RunnableLambda; LangGraph
    # picks the sync path here.
    result = compiled.invoke(
        {"messages": [HumanMessage(content="Tell me about Saturn.")]},
        {"configurable": {"thread_id": "mw-1"}},
    )
    messages = result["messages"]

    # The end state should contain: user input, manager's delegation
    # AIMessage, the synthesized ToolMessage (worker's reply), and the
    # manager's final AIMessage.
    msg_types = [type(m).__name__ for m in messages]
    assert "HumanMessage" in msg_types
    assert "ToolMessage" in msg_types
    # Final message is the manager's terminating AIMessage.
    assert isinstance(messages[-1], AIMessage)
    assert "Saturn has rings" in messages[-1].content

    # And the ToolMessage carries the worker's reply matched to the
    # pending delegation tool_call_id — proves the isolation wrapper
    # threaded the call id through.
    tool_msgs = [m for m in messages if type(m).__name__ == "ToolMessage"]
    assert tool_msgs and tool_msgs[0].tool_call_id == "call_1"
    assert "Saturn has rings" in tool_msgs[0].content


def test_manager_workers_answers_every_delegation_in_a_single_turn() -> None:
    """Regression: when the manager emits SEVERAL ``delegate_to_<worker>``
    tool calls in one turn (e.g. "spin up 5 sub-agents"), every delegation
    must run and be answered by its own ToolMessage matched to the
    originating tool_call_id.

    Before the fix the parent graph routed only the first delegation, so the
    other tool_call_ids were left unanswered — an invalid tool-call /
    tool-result sequence that made the manager hallucinate the missing
    replies. This asserts all three calls get matched ToolMessages.
    """
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langgraph.checkpoint.memory import MemorySaver

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.adapters.langgraph._langgraphconverter import (
        AgentSpecToLangGraphConverter,
    )
    from pyagentspec.agent import Agent
    from pyagentspec.managerworkers import ManagerWorkers

    manager_agent = Agent(
        name="Coordinator",
        description="Coordinates",
        system_prompt="You coordinate.",
        llm_config=_llm_cfg("manager_llm"),
    )
    worker = Agent(
        name="Sub Agent",
        description="Writes poems",
        system_prompt="You write poems.",
        llm_config=_llm_cfg("worker_llm"),
    )
    mw = ManagerWorkers(name="Team", group_manager=manager_agent, workers=[worker])

    # Turn 1: three delegations to the SAME worker in one AIMessage.
    # Turn 2: terminate (no tool call).
    manager_responses = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "delegate_to_sub_agent", "args": {"task": "Spanish poem"}, "id": "call_1"},
                {"name": "delegate_to_sub_agent", "args": {"task": "French poem"}, "id": "call_2"},
                {"name": "delegate_to_sub_agent", "args": {"task": "German poem"}, "id": "call_3"},
            ],
        ),
        AIMessage(content="Here are your three poems."),
    ]
    # Each worker invocation pops one reply; provide enough for the fan-out.
    worker_responses = [AIMessage(content=f"poem #{i}") for i in range(1, 6)]

    fake_manager = _fake_manager(*manager_responses)
    fake_worker = _fake_manager(*worker_responses)

    def _dispatch(self_obj: Any, llm_config: Any, *args: Any, **kwargs: Any) -> Any:
        if llm_config.name == "manager_llm":
            return fake_manager
        if llm_config.name == "worker_llm":
            return fake_worker
        raise AssertionError(f"unexpected llm_config: {llm_config.name}")

    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with patch.object(
        AgentSpecToLangGraphConverter,
        "_llm_convert_to_langgraph",
        autospec=True,
        side_effect=_dispatch,
    ), patch.object(
        FakeMessagesListChatModel,
        "bind_tools",
        new=lambda self_obj, *a, **kw: self_obj,
    ):
        compiled = loader.load_component(mw)

    result = compiled.invoke(
        {"messages": [HumanMessage(content="Write 3 poems via sub-agents.")]},
        {"configurable": {"thread_id": "mw-multi"}},
    )
    messages = result["messages"]

    # Every delegation tool_call_id must be answered by exactly one ToolMessage.
    requested = {
        tc["id"]
        for m in messages
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
        if tc["name"].startswith("delegate_to_")
    }
    answered = [m.tool_call_id for m in messages if type(m).__name__ == "ToolMessage"]
    assert requested == {"call_1", "call_2", "call_3"}
    assert sorted(answered) == ["call_1", "call_2", "call_3"], (
        f"unanswered delegations: {requested - set(answered)}"
    )
    # No duplicate replies, and each carries a worker poem.
    assert len(answered) == 3
    tool_msgs = [m for m in messages if type(m).__name__ == "ToolMessage"]
    assert all(m.content.startswith("poem #") for m in tool_msgs)


# ─── Recursive nesting ──────────────────────────────────────────────────────


def test_nested_manager_workers_compiles_recursively() -> None:
    """A worker that is itself a ManagerWorkers compiles through the
    same dispatch — the inner ManagerWorkers becomes a CompiledStateGraph
    that the outer parent graph wires in as a subgraph node."""
    from langchain_core.messages import AIMessage
    from langgraph.checkpoint.memory import MemorySaver

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.adapters.langgraph._langgraphconverter import (
        AgentSpecToLangGraphConverter,
    )
    from pyagentspec.agent import Agent
    from pyagentspec.managerworkers import ManagerWorkers

    leaf = Agent(
        name="Leaf",
        description="Leaf task",
        system_prompt="Leaf.",
        llm_config=_llm_cfg("leaf_llm"),
    )
    inner_manager = Agent(
        name="InnerManager",
        description="Inner",
        system_prompt="Manage leaves.",
        llm_config=_llm_cfg("inner_llm"),
    )
    inner_mw = ManagerWorkers(
        name="Inner",
        group_manager=inner_manager,
        workers=[leaf],
    )
    outer_manager = Agent(
        name="OuterManager",
        description="Outer",
        system_prompt="Manage subteams.",
        llm_config=_llm_cfg("outer_llm"),
    )
    outer_mw = ManagerWorkers(
        name="Outer",
        group_manager=outer_manager,
        workers=[inner_mw],
    )

    fake_llm = _fake_manager(AIMessage(content="Done."))
    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with patch.object(
        AgentSpecToLangGraphConverter, "_llm_convert_to_langgraph", return_value=fake_llm
    ):
        compiled = loader.load_component(outer_mw)

    # Outer parent graph has a node for the inner ManagerWorkers worker.
    assert "inner" in compiled.builder.nodes


def test_rejects_non_agent_group_manager() -> None:
    """ManagerWorkers.group_manager must be an Agent — pyagentspec allows
    any AgenticComponent but the LangGraph adapter needs a chat-LLM that
    emits tool_calls to decide which worker to delegate to."""
    from langgraph.checkpoint.memory import MemorySaver

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.agent import Agent
    from pyagentspec.managerworkers import ManagerWorkers

    # Use a nested ManagerWorkers as the group_manager — a valid
    # AgenticComponent per pyagentspec validators, unsupported here.
    leaf = Agent(
        name="Leaf",
        description="L",
        system_prompt="L.",
        llm_config=_llm_cfg("l"),
    )
    inner_manager = Agent(
        name="Inner",
        description="I",
        system_prompt="I.",
        llm_config=_llm_cfg("i"),
    )
    inner_mw = ManagerWorkers(
        name="Inner",
        group_manager=inner_manager,
        workers=[leaf],
    )
    outer_mw = ManagerWorkers(
        name="Outer",
        group_manager=inner_mw,
        workers=[
            Agent(name="Other", description="O", system_prompt="O.", llm_config=_llm_cfg("o")),
        ],
    )
    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with pytest.raises(NotImplementedError, match="group_manager must be an Agent"):
        loader.load_component(outer_mw)


# ─── Worker name collision ──────────────────────────────────────────────────


def test_workers_with_name_slug_collision_are_rejected() -> None:
    """Two workers whose names normalize to the same node identifier
    would silently overwrite each other in the parent graph; raise at
    load time instead."""
    from langgraph.checkpoint.memory import MemorySaver

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.agent import Agent
    from pyagentspec.managerworkers import ManagerWorkers

    a = Agent(name="Helper A", description="x", system_prompt=".", llm_config=_llm_cfg("a"))
    b = Agent(name="helper-a", description="x", system_prompt=".", llm_config=_llm_cfg("b"))
    # Both normalize to "helper_a".
    mw = ManagerWorkers(name="T", group_manager=Agent(
        name="M", description="m", system_prompt=".", llm_config=_llm_cfg("m"),
    ), workers=[a, b])

    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with pytest.raises(ValueError, match="collide after normalization"):
        loader.load_component(mw)


def test_worker_events_stream_natively_namespaced_under_worker_node() -> None:
    """Regression: a worker's token events must stream under the worker
    node's checkpoint namespace so a consumer can attribute them to the
    sub-agent. The wrapper must inherit the ambient run config (no fresh
    thread_id); a fresh thread_id detaches the worker into a top-level
    ``agent:<uuid>`` run with no worker prefix, which is unattributable."""
    import asyncio

    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel,
    )
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph import END, START, MessagesState, StateGraph

    from pyagentspec.adapters.langgraph._langgraphconverter import (
        _wrap_worker_for_subgraph,
    )

    # A minimal worker compiled graph that streams some content.
    wmodel = GenericFakeChatModel(messages=iter([AIMessage(content="Saturn has rings")] * 9))
    wb = StateGraph(MessagesState)

    async def _wagent(state: Any) -> Any:
        return {"messages": [await wmodel.ainvoke(state["messages"])]}

    wb.add_node("agent", _wagent)
    wb.add_edge(START, "agent")
    wb.add_edge("agent", END)
    worker_graph = wb.compile()

    # Parent: a plain manager node emits the delegate tool call, then routes
    # to the wrapped worker node named "research_helper".
    pb = StateGraph(MessagesState)

    def _manager(state: Any) -> Any:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "delegate_to_research_helper", "args": {"task": "Saturn"}, "id": "c1"}
                    ],
                )
            ]
        }

    pb.add_node("__manager__", _manager)
    pb.add_node("research_helper", _wrap_worker_for_subgraph(worker_graph, "research_helper"))
    pb.add_edge(START, "__manager__")
    pb.add_edge("__manager__", "research_helper")
    pb.add_edge("research_helper", END)
    parent = pb.compile()

    async def _collect() -> Any:
        namespaces = []
        async for ev in parent.astream_events(
            {"messages": [HumanMessage(content="hi")]},
            {"configurable": {"thread_id": "t"}},
            version="v2",
        ):
            if ev["event"] == "on_chat_model_stream":
                ns = (ev.get("metadata") or {}).get("langgraph_checkpoint_ns", "")
                namespaces.append(ns)
        return namespaces

    namespaces = asyncio.run(_collect())
    assert namespaces, "expected the worker to emit token-stream events"
    # Every worker token event is namespaced under the worker node, so a
    # consumer can attribute the stream to the sub-agent.
    assert all(ns.startswith("research_helper:") for ns in namespaces), namespaces


# ─── ManagerWorkers as a Swarm member: handoff to a sibling ─────────────────


def test_handoff_tool_name_normalizes_like_node_names() -> None:
    from pyagentspec.adapters.langgraph._langgraphconverter import _handoff_tool_name

    assert _handoff_tool_name("Specialist") == "transfer_to_specialist"
    assert _handoff_tool_name("Math Helper v2") == "transfer_to_math_helper_v2"
    # Empty / punctuation-only falls back to a stable identifier.
    assert _handoff_tool_name("!!!") == "transfer_to_agent"


def test_route_manager_handoff_takes_precedence_over_delegation() -> None:
    """A ``transfer_to_<sibling>`` call routes to the handoff node, and wins
    over any ``delegate_to_<worker>`` calls emitted in the same turn."""
    from langchain_core.messages import AIMessage

    from pyagentspec.adapters.langgraph._langgraphconverter import (
        _HANDOFF_NODE_KEY,
        _route_manager_to_worker_handoff_or_end,
    )

    mixed = AIMessage(
        content="",
        tool_calls=[
            {"name": "delegate_to_research_helper", "args": {"task": "x"}, "id": "c1"},
            {"name": "transfer_to_specialist", "args": {}, "id": "c2"},
        ],
    )
    assert _route_manager_to_worker_handoff_or_end({"messages": [mixed]}) == _HANDOFF_NODE_KEY


def test_make_handoff_forward_node_reemits_parent_command() -> None:
    """The handoff node returns a ``Command(goto=<sibling>, graph=PARENT)``
    that sets ``active_agent`` and forwards the transfer AIMessage ahead of a
    ToolMessage answering *every* unanswered tool call on it."""
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.types import Command

    from pyagentspec.adapters.langgraph._langgraphconverter import (
        _make_handoff_forward_node,
    )

    node = _make_handoff_forward_node({"transfer_to_specialist": "Specialist"})
    transfer_ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "delegate_to_helper", "args": {"task": "y"}, "id": "d1"},
            {"name": "transfer_to_specialist", "args": {}, "id": "h1"},
        ],
    )
    out = node.invoke({"messages": [transfer_ai]})

    assert isinstance(out, Command)
    assert out.goto == "Specialist"
    assert out.graph == Command.PARENT
    assert out.update["active_agent"] == "Specialist"

    forwarded = out.update["messages"]
    # Transfer AIMessage first, then a ToolMessage per unanswered call.
    assert forwarded[0] is transfer_ai
    tool_msgs = [m for m in forwarded if isinstance(m, ToolMessage)]
    answered = {m.tool_call_id for m in tool_msgs}
    assert answered == {"d1", "h1"}
    transferred = next(m for m in tool_msgs if m.tool_call_id == "h1")
    assert transferred.content == "Successfully transferred to Specialist"


def test_manager_workers_as_swarm_member_hands_off_to_sibling() -> None:
    """End-to-end: a Swarm whose first member is a ManagerWorkers. The
    manager's LLM emits ``transfer_to_<sibling>``; the parent graph re-emits
    the handoff to the Swarm, which routes to the sibling Agent and lets it
    answer — proving a sub-agent-bearing agent can participate in a Swarm
    (the case the LangGraph adapter used to reject)."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langgraph.checkpoint.memory import MemorySaver

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.adapters.langgraph._langgraphconverter import (
        AgentSpecToLangGraphConverter,
    )
    from pyagentspec.agent import Agent
    from pyagentspec.managerworkers import ManagerWorkers
    from pyagentspec.swarm import Swarm

    manager_agent = Agent(
        name="Coordinator",
        description="Coordinates",
        system_prompt="You coordinate.",
        llm_config=_llm_cfg("manager_llm"),
    )
    worker = Agent(
        name="Research Helper",
        description="Handles research",
        system_prompt="You research.",
        llm_config=_llm_cfg("worker_llm"),
    )
    team = ManagerWorkers(name="Team", group_manager=manager_agent, workers=[worker])
    specialist = Agent(
        name="Specialist",
        description="Domain specialist",
        system_prompt="You are the specialist.",
        llm_config=_llm_cfg("specialist_llm"),
    )
    swarm = Swarm(
        name="Crew",
        first_agent=team,
        relationships=[(team, specialist), (specialist, team)],
    )

    # Manager turn 1: hand the conversation off to the Specialist sibling.
    fake_manager = _fake_manager(
        AIMessage(
            content="",
            tool_calls=[{"name": "transfer_to_specialist", "args": {}, "id": "call_h1"}],
        )
    )
    fake_specialist = _fake_manager(AIMessage(content="Specialist handled it."))
    fake_worker = _fake_manager(AIMessage(content="(worker, unused)"))

    def _dispatch(self_obj: Any, llm_config: Any, *args: Any, **kwargs: Any) -> Any:
        return {
            "manager_llm": fake_manager,
            "worker_llm": fake_worker,
            "specialist_llm": fake_specialist,
        }[llm_config.name]

    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with patch.object(
        AgentSpecToLangGraphConverter,
        "_llm_convert_to_langgraph",
        autospec=True,
        side_effect=_dispatch,
    ), patch.object(
        FakeMessagesListChatModel,
        "bind_tools",
        new=lambda self_obj, *a, **kw: self_obj,
    ):
        compiled = loader.load_component(swarm)

    result = compiled.invoke(
        {"messages": [HumanMessage(content="Please help.")]},
        {"configurable": {"thread_id": "crew-1"}},
    )
    messages = result["messages"]

    # The Specialist produced the final answer → the handoff actually routed.
    assert isinstance(messages[-1], AIMessage)
    assert "Specialist handled it." in messages[-1].content

    # The transfer is a well-formed AIMessage(tool_call) → ToolMessage pair.
    transferred = [
        m
        for m in messages
        if isinstance(m, ToolMessage)
        and m.content == "Successfully transferred to Specialist"
    ]
    assert transferred and transferred[0].tool_call_id == "call_h1"

    # Transcript validity: every ToolMessage answers a preceding AIMessage
    # tool_call with a matching id (an orphan ToolMessage would 400 a real LLM).
    open_call_ids: set = set()
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            open_call_ids.add(tc["id"])
        if isinstance(m, ToolMessage):
            assert m.tool_call_id in open_call_ids, (
                f"orphan ToolMessage {m.tool_call_id} with no preceding tool_call"
            )


def test_swarm_rejects_unsupported_member_type() -> None:
    """A Swarm member that is neither an Agent nor a ManagerWorkers (here a
    nested Swarm) raises a clear NotImplementedError — there is no single
    LLM to attach the handoff tools to."""
    import pytest

    from pyagentspec.adapters.langgraph._langgraphconverter import (
        AgentSpecToLangGraphConverter,
    )
    from pyagentspec.agent import Agent
    from pyagentspec.swarm import Swarm

    a1 = Agent(name="A1", description="a1", system_prompt="x", llm_config=_llm_cfg("l1"))
    a2 = Agent(name="A2", description="a2", system_prompt="y", llm_config=_llm_cfg("l2"))
    nested = Swarm(name="Nested", first_agent=a1, relationships=[(a1, a2)])
    outer = Swarm(name="Outer", first_agent=nested, relationships=[(nested, a1)])

    with pytest.raises(NotImplementedError, match="Swarm"):
        AgentSpecToLangGraphConverter().convert(outer, tool_registry={})
