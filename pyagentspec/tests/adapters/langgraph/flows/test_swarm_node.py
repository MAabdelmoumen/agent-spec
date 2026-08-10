# Copyright © 2026 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.

"""A Swarm used as a flow step (AgentNode).

Regression coverage for two coupled behaviours, symmetric with the ManagerWorkers
flow-step coverage in ``test_managerworkers_node.py``:
  * ``Swarm._get_inferred_inputs`` exposes the entry agent's (``first_agent``) inputs, so a
    flow ``AgentNode`` wrapping a swarm declares input ports and a ``DataFlowEdge`` into it
    resolves at load (previously: "node does not have any input property...").
  * ``AgentNodeExecutor`` runs a Swarm node (previously: TypeError "can only be used with
    AgentSpecAgent agents"), rendering the node inputs into the entry agent's prompt and
    returning its result.
"""

from pyagentspec.agent import Agent
from pyagentspec.property import StringProperty
from pyagentspec.swarm import Swarm


def test_swarm_infers_inputs_from_first_agent_prompt() -> None:
    """A Swarm exposes the entry agent's prompt placeholders as inputs."""
    from pyagentspec.llms.openaicompatibleconfig import OpenAiCompatibleConfig

    cfg = OpenAiCompatibleConfig(name="m", model_id="fake", url="null")
    first = Agent(
        name="first",
        llm_config=cfg,
        system_prompt="Translate the following to Arabic:\n\n{{joke}}\n\nMake {{count}} variants.",
    )
    second = Agent(name="second", llm_config=cfg, system_prompt="You translate.")
    swarm = Swarm(name="swarm", first_agent=first, relationships=[(first, second)])

    assert sorted(p.title for p in (swarm.inputs or [])) == ["count", "joke"]


def test_swarm_runs_as_a_flow_step_with_data_edge_inputs() -> None:
    """A Swarm flow step loads (data edge resolves) and executes offline.

    The model is stubbed (no real LLM, no handoff), so the entry agent produces a final
    message and the swarm routes straight to END. Asserts the flow both loads — proving the
    swarm node exposes the ``joke`` input the data edge targets — and runs, surfacing the
    entry agent's answer as the node's single string output.
    """
    from unittest.mock import patch

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import MemorySaver

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.adapters.langgraph._langgraphconverter import (
        AgentSpecToLangGraphConverter,
    )
    from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
    from pyagentspec.flows.flow import Flow
    from pyagentspec.flows.nodes import AgentNode, EndNode, StartNode
    from pyagentspec.llms.openaicompatibleconfig import OpenAiCompatibleConfig

    class _FakeModel(FakeMessagesListChatModel, ChatOpenAI):
        pass

    # Final message has no tool_calls → the entry agent answers without handing off.
    fake_llm = _FakeModel(responses=[AIMessage(content="لماذا...")])

    cfg = OpenAiCompatibleConfig(name="agent_llm", model_id="fake", url="null")
    joke = StringProperty(title="joke")
    translated = StringProperty(title="translated")

    first = Agent(
        name="first",
        llm_config=cfg,
        system_prompt="Translate the following to Arabic:\n\n{{joke}}",
        outputs=[translated],
    )
    second = Agent(name="second", llm_config=cfg, system_prompt="You translate.")
    swarm = Swarm(name="translator", first_agent=first, relationships=[(first, second)])
    # The swarm node exposes the entry agent's `joke` input, and its single `translated`
    # output (inherited from the entry agent) for the leaf edge.
    assert [p.title for p in (swarm.inputs or [])] == ["joke"]

    swarm_node = AgentNode(name="swarm_node", agent=swarm)
    start_node = StartNode(name="start", inputs=[joke])
    end_node = EndNode(name="end", outputs=[translated])
    flow = Flow(
        name="flow",
        start_node=start_node,
        nodes=[start_node, swarm_node, end_node],
        control_flow_connections=[
            ControlFlowEdge(name="start_to_node", from_node=start_node, to_node=swarm_node),
            ControlFlowEdge(name="node_to_end", from_node=swarm_node, to_node=end_node),
        ],
        data_flow_connections=[
            DataFlowEdge(
                name="joke_edge",
                source_node=start_node,
                source_output=joke.title,
                destination_node=swarm_node,
                destination_input=joke.title,
            ),
            DataFlowEdge(
                name="translated_edge",
                source_node=swarm_node,
                source_output=translated.title,
                destination_node=end_node,
                destination_input=translated.title,
            ),
        ],
        outputs=[translated],
    )

    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with patch.object(
        AgentSpecToLangGraphConverter,
        "_llm_convert_to_langgraph",
        autospec=True,
        side_effect=lambda self_obj, llm_config, *a, **k: fake_llm,
    ), patch.object(
        FakeMessagesListChatModel,
        "bind_tools",
        new=lambda self_obj, *a, **k: self_obj,
    ):
        compiled = loader.load_component(flow)
        result = compiled.invoke(
            {
                "inputs": {"joke": "Why did the car..."},
                "messages": [{"role": "user", "content": ""}],
            },
            {"configurable": {"thread_id": "swarm-node"}},
        )

    assert "outputs" in result
    assert result["outputs"]["translated"] == "لماذا..."


def test_swarm_node_resolves_multiple_structured_outputs() -> None:
    """A Swarm flow step resolves several structured outputs.

    Symmetric with ``test_managerworkers_node_resolves_multiple_structured_outputs``: a
    swarm also runs over ``MessagesState`` and returns no ``structured_response``, so two
    or more declared outputs used to raise ``ValueError`` at the producing node. The
    generated values are recovered from the structured-output tool call in the history.
    """
    from unittest.mock import patch

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import MemorySaver

    from pyagentspec.adapters.langgraph import AgentSpecLoader
    from pyagentspec.adapters.langgraph._langgraphconverter import (
        AgentSpecToLangGraphConverter,
    )
    from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
    from pyagentspec.flows.flow import Flow
    from pyagentspec.flows.nodes import AgentNode, EndNode, StartNode
    from pyagentspec.llms.openaicompatibleconfig import OpenAiCompatibleConfig

    class _FakeModel(FakeMessagesListChatModel, ChatOpenAI):
        pass

    fake_llm = _FakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "AgentOutputModel",
                        "args": {"route": "art", "brief": "a pixel dragon"},
                        "id": "call_structured_output",
                    }
                ],
            )
        ]
    )

    cfg = OpenAiCompatibleConfig(name="agent_llm", model_id="fake", url="null")
    request = StringProperty(title="request")
    route = StringProperty(title="route")
    brief = StringProperty(title="brief")

    first = Agent(
        name="first",
        llm_config=cfg,
        system_prompt="Triage this request:\n\n{{request}}",
        outputs=[route, brief],
    )
    second = Agent(name="second", llm_config=cfg, system_prompt="You help.")
    swarm = Swarm(name="triage", first_agent=first, relationships=[(first, second)])

    swarm_node = AgentNode(name="swarm_node", agent=swarm)
    start_node = StartNode(name="start", inputs=[request])
    end_node = EndNode(name="end", outputs=[route, brief])
    flow = Flow(
        name="flow",
        start_node=start_node,
        nodes=[start_node, swarm_node, end_node],
        control_flow_connections=[
            ControlFlowEdge(name="start_to_node", from_node=start_node, to_node=swarm_node),
            ControlFlowEdge(name="node_to_end", from_node=swarm_node, to_node=end_node),
        ],
        data_flow_connections=[
            DataFlowEdge(
                name="request_edge",
                source_node=start_node,
                source_output=request.title,
                destination_node=swarm_node,
                destination_input=request.title,
            ),
            DataFlowEdge(
                name="route_edge",
                source_node=swarm_node,
                source_output=route.title,
                destination_node=end_node,
                destination_input=route.title,
            ),
            DataFlowEdge(
                name="brief_edge",
                source_node=swarm_node,
                source_output=brief.title,
                destination_node=end_node,
                destination_input=brief.title,
            ),
        ],
        outputs=[route, brief],
    )

    loader = AgentSpecLoader(tool_registry={}, checkpointer=MemorySaver())
    with patch.object(
        AgentSpecToLangGraphConverter,
        "_llm_convert_to_langgraph",
        autospec=True,
        side_effect=lambda self_obj, llm_config, *a, **k: fake_llm,
    ), patch.object(
        FakeMessagesListChatModel,
        "bind_tools",
        new=lambda self_obj, *a, **k: self_obj,
    ):
        compiled = loader.load_component(flow)
        result = compiled.invoke(
            {
                "inputs": {"request": "make me a dragon"},
                "messages": [{"role": "user", "content": ""}],
            },
            {"configurable": {"thread_id": "swarm-node-structured"}},
        )

    assert result["outputs"]["route"] == "art"
    assert result["outputs"]["brief"] == "a pixel dragon"
