# Copyright © 2025, 2026 Oracle and/or its affiliates.
#
# This software is under the Apache License 2.0
# (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0) or Universal Permissive License
# (UPL) 1.0 (LICENSE-UPL or https://oss.oracle.com/licenses/upl), at your option.


import inspect
import logging
import re
import sys
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    TypeGuard,
    Union,
    cast,
    overload,
)
from uuid import uuid4

from pydantic import BaseModel, SecretStr
from typing_extensions import NotRequired, Required, TypedDict

from pyagentspec import Component as AgentSpecComponent
from pyagentspec.adapters._tools_common import _create_remote_tool_func
from pyagentspec.adapters._utils import (
    SchemaRegistry,
    _build_type_from_schema,
    create_pydantic_model_from_properties,
)
from pyagentspec.adapters.langgraph._node_execution import (
    NodeExecutor,
    extract_outputs_from_invoke_result,
    is_single_string_output,
)
from pyagentspec.adapters.langgraph._types import (
    AgentState,
    BaseCallbackHandler,
    BaseChatModel,
    BaseTool,
    Checkpointer,
    CompiledStateGraph,
    ControlFlow,
    FlowInputSchema,
    FlowOutputSchema,
    FlowStateSchema,
    LangGraphTool,
    RunnableConfig,
    StateGraph,
    StructuredTool,
    interrupt,
    langchain_agents,
    langgraph_graph,
    langgraph_swarm,
)
from pyagentspec.adapters.langgraph.mcp_utils import _HttpxClientFactory, run_async_in_sync
from pyagentspec.adapters.langgraph.tracing import (
    AgentSpecLlmCallbackHandler,
    AgentSpecToolCallbackHandler,
)
from pyagentspec.agent import Agent as AgentSpecAgent
from pyagentspec.flows.edges import ControlFlowEdge as AgentSpecControlFlowEdge
from pyagentspec.flows.edges import DataFlowEdge as AgentSpecDataFlowEdge
from pyagentspec.flows.flow import Flow as AgentSpecFlow
from pyagentspec.flows.node import Node as AgentSpecNode
from pyagentspec.flows.nodes import AgentNode as AgentSpecAgentNode
from pyagentspec.flows.nodes import ApiNode as AgentSpecApiNode
from pyagentspec.flows.nodes import BranchingNode as AgentSpecBranchingNode
from pyagentspec.flows.nodes import CatchExceptionNode as AgentSpecCatchExceptionNode
from pyagentspec.flows.nodes import EndNode as AgentSpecEndNode
from pyagentspec.flows.nodes import FlowNode as AgentSpecFlowNode
from pyagentspec.flows.nodes import InputMessageNode as AgentSpecInputMessageNode
from pyagentspec.flows.nodes import LlmNode as AgentSpecLlmNode
from pyagentspec.flows.nodes import MapNode as AgentSpecMapNode
from pyagentspec.flows.nodes import OutputMessageNode as AgentSpecOutputMessageNode
from pyagentspec.flows.nodes import StartNode as AgentSpecStartNode
from pyagentspec.flows.nodes import ToolNode as AgentSpecToolNode
from pyagentspec.llms.llmconfig import LlmConfig as AgentSpecLlmConfig
from pyagentspec.llms.llmgenerationconfig import LlmGenerationConfig
from pyagentspec.llms.ociclientconfig import (
    OciClientConfig,
    OciClientConfigWithApiKey,
    OciClientConfigWithInstancePrincipal,
    OciClientConfigWithResourcePrincipal,
    OciClientConfigWithSecurityToken,
)
from pyagentspec.llms.ocigenaiconfig import OciGenAiConfig
from pyagentspec.llms.ollamaconfig import OllamaConfig
from pyagentspec.llms.openaicompatibleconfig import OpenAIAPIType, OpenAiCompatibleConfig
from pyagentspec.llms.openaiconfig import OpenAiConfig
from pyagentspec.llms.vllmconfig import VllmConfig
from pyagentspec.managerworkers import ManagerWorkers as AgentSpecManagerWorkers
from pyagentspec.mcp.clienttransport import ClientTransport as AgentSpecClientTransport
from pyagentspec.mcp.clienttransport import RemoteTransport as AgentSpecRemoteTransport
from pyagentspec.mcp.clienttransport import SSEmTLSTransport as AgentSpecSSEmTLSTransport
from pyagentspec.mcp.clienttransport import SSETransport as AgentSpecSSETransport
from pyagentspec.mcp.clienttransport import StdioTransport as AgentSpecStdioTransport
from pyagentspec.mcp.clienttransport import (
    StreamableHTTPmTLSTransport as AgentSpecStreamableHTTPmTLSTransport,
)
from pyagentspec.mcp.clienttransport import (
    StreamableHTTPTransport as AgentSpecStreamableHTTPTransport,
)
from pyagentspec.mcp.tools import MCPTool as AgentSpecMCPTool
from pyagentspec.mcp.tools import MCPToolBox as AgentSpecMCPToolBox
from pyagentspec.mcp.tools import MCPToolSpec as AgentSpecMCPToolSpec
from pyagentspec.property import ListProperty as AgentSpecListProperty
from pyagentspec.property import Property as AgentSpecProperty
from pyagentspec.property import StringProperty as AgentSpecStringProperty
from pyagentspec.property import _empty_default as _agentspec_empty_default
from pyagentspec.property import json_schemas_have_same_type
from pyagentspec.retrypolicy import RetryPolicy
from pyagentspec.swarm import HandoffMode as AgentSpecHandoffMode
from pyagentspec.swarm import Swarm as AgentSpecSwarm
from pyagentspec.tools import ClientTool as AgentSpecClientTool
from pyagentspec.tools import RemoteTool as AgentSpecRemoteTool
from pyagentspec.tools import ServerTool as AgentSpecServerTool
from pyagentspec.tools import Tool as AgentSpecTool
from pyagentspec.tools import ToolBox as AgentSpecToolBox
from pyagentspec.tracing.events import AgentExecutionEnd as AgentSpecAgentExecutionEnd
from pyagentspec.tracing.events import AgentExecutionStart as AgentSpecAgentExecutionStart
from pyagentspec.tracing.events import FlowExecutionEnd as AgentSpecFlowExecutionEnd
from pyagentspec.tracing.events import FlowExecutionStart as AgentSpecFlowExecutionStart
from pyagentspec.tracing.events import (
    ManagerWorkersExecutionEnd as AgentSpecManagerWorkersExecutionEnd,
)
from pyagentspec.tracing.events import (
    ManagerWorkersExecutionStart as AgentSpecManagerWorkersExecutionStart,
)
from pyagentspec.tracing.spans import AgentExecutionSpan as AgentSpecAgentExecutionSpan
from pyagentspec.tracing.spans import FlowExecutionSpan as AgentSpecFlowExecutionSpan
from pyagentspec.tracing.spans import (
    ManagerWorkersExecutionSpan as AgentSpecManagerWorkersExecutionSpan,
)

if TYPE_CHECKING:
    from langchain_mcp_adapters.sessions import (
        SSEConnection,
        StdioConnection,
        StreamableHttpConnection,
    )


def _mcp_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("langchain_mcp_adapters") is not None


def _ensure_mcp_dependency_installed() -> None:
    if not _mcp_available():
        raise RuntimeError(
            "langchain-mcp-adapters is required to preload MCP tools. "
            "Install it (e.g., pip install langchain-mcp-adapters) or remove MCP tools from the spec."
        )


def _create_agent_state_typed_dict(
    model_name: str,
    inputs: List[AgentSpecProperty],
) -> "type[AgentState[BaseModel]]":
    """Create a TypedDict subclass of LangChain's AgentState with custom inputs.

    We extend the default AgentState (which already includes `messages` and
    optional `structured_response`) by adding our input properties and a
    required `remaining_steps` field. Required/optional inputs are expressed
    using PEP 655 `Required`/`NotRequired`.
    """
    import types

    registry = SchemaRegistry()

    annotations: Dict[str, Any] = {
        "remaining_steps": Required[int],
    }

    for property_ in inputs:
        annotation = _build_type_from_schema(property_.title, property_.json_schema, registry)
        if property_.default is not _agentspec_empty_default:
            annotations[property_.title] = NotRequired[annotation]
        else:
            annotations[property_.title] = Required[annotation]

    def _exec_body(ns: Dict[str, Any]) -> None:
        ns["__annotations__"] = annotations

    # total=False => unspecified fields are optional unless wrapped with Required
    return types.new_class(
        model_name,
        (AgentState[BaseModel],),
        {"total": False},
        _exec_body,
    )


def _remote_transport_headers(
    transport: AgentSpecRemoteTransport,
) -> Optional[Dict[str, str]]:
    """Return the headers to send on the wire for a remote MCP transport.

    A ``RemoteTransport`` carries both ``headers`` and ``sensitive_headers``,
    validated to be disjoint. ``sensitive_headers`` is only redacted from
    *exported* configs (so credentials never leak into a saved spec) -- it must
    still travel on live requests. Merge both so a header configured as
    sensitive (e.g. an ``Authorization`` token) actually reaches the server.
    """
    merged = {**(transport.headers or {}), **(transport.sensitive_headers or {})}
    return merged or None


class AgentSpecToLangGraphConverter:
    def convert(
        self,
        agentspec_component: AgentSpecComponent,
        tool_registry: Dict[str, LangGraphTool],
        converted_components: Optional[Dict[str, Any]] = None,
        checkpointer: Optional[Checkpointer] = None,
        config: Optional[RunnableConfig] = None,
        middleware: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Convert the given PyAgentSpec component object into the corresponding LangGraph component.

        Parameters
        ----------
        agentspec_component:
            The Agent Spec component to convert.
        tool_registry:
            Dictionary mapping tool names to LangGraph tool objects.
        converted_components:
            Optional cache of already-converted components (keyed by component id).
        checkpointer:
            Optional LangGraph checkpointer to wire into created graphs.
        config:
            Optional ``RunnableConfig`` to pass to created runnables/graphs.
        middleware:
            Optional list of LangChain agent middleware instances forwarded to
            ``langchain_agents.create_agent(middleware=...)`` when compiling an Agent
            Spec ``Agent`` into a ReAct graph. Order is preserved — index ``0`` is the
            outermost middleware. When ``None`` or an empty list, the ``middleware``
            keyword is omitted entirely from the ``create_agent`` call.
        """
        middleware_list: List[Any] = list(middleware or [])
        if converted_components is None:
            converted_components = {}
        if config is None:
            if checkpointer is not None:
                config = RunnableConfig({"configurable": {"thread_id": str(uuid4())}})
            else:
                config = RunnableConfig({})
        if agentspec_component.id not in converted_components:
            converted_components[agentspec_component.id] = self._convert(
                agentspec_component,
                tool_registry,
                converted_components,
                checkpointer,
                config,
                middleware_list,
            )
        return converted_components[agentspec_component.id]

    def _convert(
        self,
        agentspec_component: AgentSpecComponent,
        tool_registry: Dict[str, LangGraphTool],
        converted_components: Dict[str, Any],
        checkpointer: Optional[Checkpointer],
        config: RunnableConfig,
        middleware: List[Any],
    ) -> Any:
        if isinstance(agentspec_component, AgentSpecAgent):
            return self._agent_convert_to_langgraph(
                agentspec_component,
                tool_registry=tool_registry,
                converted_components=converted_components,
                checkpointer=checkpointer,
                config=config,
                middleware=middleware,
            )
        elif isinstance(agentspec_component, AgentSpecSwarm):
            return self._swarm_convert_to_langgraph(
                agentspec_component,
                tool_registry=tool_registry,
                converted_components=converted_components,
                checkpointer=checkpointer,
                config=config,
                middleware=middleware,
            )
        elif isinstance(agentspec_component, AgentSpecManagerWorkers):
            return self._manager_workers_convert_to_langgraph(
                agentspec_component,
                tool_registry=tool_registry,
                converted_components=converted_components,
                checkpointer=checkpointer,
                config=config,
                middleware=middleware,
            )
        elif isinstance(agentspec_component, AgentSpecLlmConfig):
            return self._llm_convert_to_langgraph(agentspec_component, config=config)
        elif isinstance(agentspec_component, AgentSpecClientTransport):
            return self._client_transport_convert_to_langgraph(agentspec_component)
        elif isinstance(agentspec_component, AgentSpecMCPTool):
            _ensure_mcp_dependency_installed()
            _ensure_checkpointer_and_valid_tool_config(agentspec_component, checkpointer)
            return self._mcp_tool_convert_to_langgraph(
                agentspec_component,
                tool_registry=tool_registry,
                converted_components=converted_components,
            )
        elif isinstance(agentspec_component, AgentSpecMCPToolBox):
            _ensure_mcp_dependency_installed()
            return self._mcp_toolbox_convert_to_langgraph(
                agentspec_component,
                tool_registry=tool_registry,
                converted_components=converted_components,
            )
        elif isinstance(agentspec_component, AgentSpecServerTool):
            _ensure_checkpointer_and_valid_tool_config(agentspec_component, checkpointer)
            return self._server_tool_convert_to_langgraph(
                agentspec_component, tool_registry, config=config
            )
        elif isinstance(agentspec_component, AgentSpecClientTool):
            _ensure_checkpointer_and_valid_tool_config(agentspec_component, checkpointer)
            return self._client_tool_convert_to_langgraph(agentspec_component)
        elif isinstance(agentspec_component, AgentSpecRemoteTool):
            _ensure_checkpointer_and_valid_tool_config(agentspec_component, checkpointer)
            return self._remote_tool_convert_to_langgraph(agentspec_component, config=config)
        elif isinstance(agentspec_component, AgentSpecFlow):
            return self._flow_convert_to_langgraph(
                agentspec_component,
                tool_registry=tool_registry,
                converted_components=converted_components,
                checkpointer=checkpointer,
                config=config,
                middleware=middleware,
            )
        elif isinstance(agentspec_component, AgentSpecNode):
            return self._node_convert_to_langgraph(
                agentspec_component,
                tool_registry=tool_registry,
                converted_components=converted_components,
                checkpointer=checkpointer,
                config=config,
                middleware=middleware,
            )
        elif isinstance(agentspec_component, AgentSpecComponent):
            raise NotImplementedError(
                f"The Agent Spec type '{agentspec_component.__class__.__name__}' is not yet supported for conversion."
            )
        else:
            raise TypeError(
                f"Expected object of type 'pyagentspec.component.Component',"
                f" but got {type(agentspec_component)} instead"
            )

    def _create_control_flow(
        self, control_flow_connections: List[AgentSpecControlFlowEdge]
    ) -> "ControlFlow":
        control_flow: "ControlFlow" = {}
        for control_flow_edge in control_flow_connections:
            source_node_id = control_flow_edge.from_node.id
            if source_node_id not in control_flow:
                control_flow[source_node_id] = {}

            branch_name = control_flow_edge.from_branch or AgentSpecNode.DEFAULT_NEXT_BRANCH
            control_flow[source_node_id][branch_name] = control_flow_edge.to_node.id

        return control_flow

    def _add_conditional_edges_to_graph(
        self,
        control_flow: "ControlFlow",
        graph_builder: StateGraph["FlowStateSchema", None, "FlowInputSchema", "FlowOutputSchema"],
    ) -> None:
        for source_node_id, control_flow_mapping in control_flow.items():
            get_branch = lambda state: state["node_execution_details"].get(
                "branch", AgentSpecNode.DEFAULT_NEXT_BRANCH
            )
            graph_builder.add_conditional_edges(source_node_id, get_branch, control_flow_mapping)

    def _flow_convert_to_langgraph(
        self,
        flow: AgentSpecFlow,
        tool_registry: Dict[str, "LangGraphTool"],
        converted_components: Dict[str, Any],
        checkpointer: Optional[Checkpointer],
        config: RunnableConfig,
        middleware: List[Any],
    ) -> CompiledStateGraph[Any, Any, Any]:

        graph_builder = StateGraph(
            FlowStateSchema, input_schema=FlowInputSchema, output_schema=FlowOutputSchema
        )

        graph_builder.add_edge(langgraph_graph.START, flow.start_node.id)

        node_executors = {
            node.id: self.convert(
                node,
                tool_registry=tool_registry,
                converted_components=converted_components,
                checkpointer=checkpointer,
                config=config,
                middleware=middleware,
            )
            for node in flow.nodes
        }

        def _find_property(properties: List[AgentSpecProperty], name: str) -> AgentSpecProperty:
            return next((property_ for property_ in properties if property_.title == name))

        # We tell the MapNodes which inputs they should iterate over
        # Based on the type of the outputs they are connected to
        for agentspec_node in flow.nodes:
            if isinstance(agentspec_node, AgentSpecMapNode):
                inputs_to_iterate = []
                for data_flow_edge in flow.data_flow_connections or []:
                    if data_flow_edge.destination_node is agentspec_node:
                        source_property = _find_property(
                            data_flow_edge.source_node.outputs or [],
                            data_flow_edge.source_output,
                        )
                        inner_flow_input_property = _find_property(
                            agentspec_node.subflow.inputs or [],
                            data_flow_edge.destination_input.replace("iterated_", "", 1),
                        )
                        if json_schemas_have_same_type(
                            source_property.json_schema,
                            AgentSpecListProperty(item_type=inner_flow_input_property).json_schema,
                        ):
                            inputs_to_iterate.append(data_flow_edge.destination_input)
                node_executors[agentspec_node.id].set_inputs_to_iterate(inputs_to_iterate)
            elif isinstance(agentspec_node, AgentSpecEndNode):
                node_executors[agentspec_node.id].set_flow_outputs(flow.outputs)

        from pyagentspec.adapters.langgraph._types import RunnableLambda

        for node_id, node_executor in node_executors.items():
            # Provide both sync and async entrypoints natively. LangGraph will use
            # the appropriate one based on invoke/stream vs ainvoke/astream.
            runnable = RunnableLambda(
                func=node_executor,
                afunc=node_executor.__acall__,
                name=node_id,
            )
            graph_builder.add_node(node_id, runnable)

        data_flow_connections: List[AgentSpecDataFlowEdge] = []
        if flow.data_flow_connections is None:
            # We manually create data flow connections if they are not given in the flow
            # This is the conversion recommended in the Agent Spec language specification
            for source_node in flow.nodes:
                for destination_node in flow.nodes:
                    for source_output in source_node.outputs or []:
                        for destination_input in destination_node.inputs or []:
                            if source_output.title == destination_input.title:
                                data_flow_connections.append(
                                    AgentSpecDataFlowEdge(
                                        name=f"{source_node.name}-{destination_node.name}-{source_output.title}",
                                        source_node=source_node,
                                        source_output=source_output.title,
                                        destination_node=destination_node,
                                        destination_input=destination_input.title,
                                    )
                                )
        else:
            data_flow_connections = flow.data_flow_connections

        for data_flow_edge in data_flow_connections:
            node_executors[data_flow_edge.source_node.id].attach_edge(data_flow_edge)

        control_flow: "ControlFlow" = self._create_control_flow(flow.control_flow_connections)
        self._add_conditional_edges_to_graph(control_flow, graph_builder)
        compiled_graph = graph_builder.compile(checkpointer=checkpointer)

        # Warn users on Python < 3.11 about async interrupts potentially failing.
        # We warn during graph loading/compilation time to avoid runtime surprises.
        if sys.version_info < (3, 11):
            uses_interrupt = False
            try:
                # Any InputMessageNode implies interrupt; client tools may interrupt too.
                uses_interrupt = any(
                    isinstance(n, AgentSpecInputMessageNode) or isinstance(n, AgentSpecToolNode)
                    for n in flow.nodes
                )
            except Exception:
                uses_interrupt = False
            if uses_interrupt or checkpointer is not None:
                logger = logging.getLogger("pyagentspec.adapters.langgraph")
                logger.warning(
                    "Async interrupts on Python < 3.11 may raise 'Called get_config outside of a runnable context'. "
                    "Prefer invoke/stream or upgrade to Python 3.11+ for ainvoke/astream."
                )

        # To enable flow execution traces monkey patch all the functions that invoke the compiled graph

        original_stream = compiled_graph.stream

        def patch_with_flow_execution_span(*args: Any, **kwargs: Any) -> Generator[Any, Any, None]:
            span_name = f"FlowExecution[{flow.name}]"
            inputs = kwargs.get("input", {})
            if not isinstance(inputs, dict):
                inputs = {}
            with AgentSpecFlowExecutionSpan(name=span_name, flow=flow) as span:
                span.add_event(AgentSpecFlowExecutionStart(flow=flow, inputs=inputs))
                original_result: dict[str, Any] | Any = {}
                result: dict[str, Any]
                # This is going to patch stream and astream, that return iterators and yield chunks
                for chunk in original_stream(*args, **kwargs):
                    yield chunk
                    if isinstance(chunk, tuple):
                        original_result = chunk[1]
                if not isinstance(original_result, dict):
                    result = {}
                else:
                    result = original_result
                span.add_event(
                    AgentSpecFlowExecutionEnd(
                        flow=flow,
                        outputs=result.get("outputs", {}),
                        branch_selected=result.get("node_execution_details", {}).get("branch", ""),
                    )
                )

        original_astream = compiled_graph.astream

        async def patch_async_with_flow_execution_span(
            *args: Any, **kwargs: Any
        ) -> AsyncGenerator[Any, Any]:
            span_name = f"FlowExecution[{flow.name}]"
            inputs = kwargs.get("input", {})
            if not isinstance(inputs, dict):
                inputs = {}
            span = AgentSpecFlowExecutionSpan(name=span_name, flow=flow)
            try:
                await span.start_async()
            except NotImplementedError:
                span.start()
            try:
                try:
                    await span.add_event_async(
                        AgentSpecFlowExecutionStart(flow=flow, inputs=inputs)
                    )
                except NotImplementedError:
                    span.add_event(AgentSpecFlowExecutionStart(flow=flow, inputs=inputs))
                original_result: dict[str, Any] | Any = {}
                result: dict[str, Any]
                # This is going to patch stream and astream, that return iterators and yield chunks
                async for chunk in original_astream(*args, **kwargs):
                    yield chunk
                    if isinstance(chunk, tuple):
                        original_result = chunk[1]
                if not isinstance(original_result, dict):
                    result = {}
                else:
                    result = original_result
                span_end_event = AgentSpecFlowExecutionEnd(
                    flow=flow,
                    outputs=result.get("outputs", {}),
                    branch_selected=result.get("node_execution_details", {}).get("branch", ""),
                )
                try:
                    await span.add_event_async(span_end_event)
                except NotImplementedError:
                    span.add_event(span_end_event)
            finally:
                try:
                    await span.end_async()
                except NotImplementedError:
                    span.end()

        # Monkey patch invocation functions to inject tracing
        # No need to patch `(a)invoke` as the internally use `(a)stream`
        compiled_graph.stream = patch_with_flow_execution_span  # type: ignore
        compiled_graph.astream = patch_async_with_flow_execution_span  # type: ignore
        return compiled_graph

    def _node_convert_to_langgraph(
        self,
        node: AgentSpecNode,
        tool_registry: Dict[str, "LangGraphTool"],
        converted_components: Dict[str, Any],
        checkpointer: Optional[Checkpointer],
        config: RunnableConfig,
        middleware: List[Any],
    ) -> "NodeExecutor":
        if isinstance(node, AgentSpecStartNode):
            return self._start_node_convert_to_langgraph(node)
        elif isinstance(node, AgentSpecEndNode):
            return self._end_node_convert_to_langgraph(node)
        elif isinstance(node, AgentSpecToolNode):
            return self._tool_node_convert_to_langgraph(
                node,
                tool_registry=tool_registry,
                converted_components=converted_components,
                checkpointer=checkpointer,
                config=config,
            )
        elif isinstance(node, AgentSpecLlmNode):
            return self._llm_node_convert_to_langgraph(
                node,
                tool_registry=tool_registry,
                converted_components=converted_components,
                checkpointer=checkpointer,
                config=config,
            )
        elif isinstance(node, AgentSpecAgentNode):
            return self._agent_node_convert_to_langgraph(
                node,
                tool_registry=tool_registry,
                converted_components=converted_components,
                checkpointer=checkpointer,
                config=config,
                middleware=middleware,
            )
        elif isinstance(node, AgentSpecBranchingNode):
            return self._branching_node_convert_to_langgraph(node)
        elif isinstance(node, AgentSpecApiNode):
            return self._api_node_convert_to_langgraph(node)
        elif isinstance(node, AgentSpecFlowNode):
            return self._flow_node_convert_to_langgraph(
                node,
                tool_registry=tool_registry,
                converted_components=converted_components,
                checkpointer=checkpointer,
                config=config,
                middleware=middleware,
            )
        elif isinstance(node, AgentSpecCatchExceptionNode):
            return self._catch_exception_node_convert_to_langgraph(
                node,
                tool_registry=tool_registry,
                converted_components=converted_components,
                checkpointer=checkpointer,
                config=config,
                middleware=middleware,
            )
        elif isinstance(node, AgentSpecInputMessageNode):
            return self._input_message_node_convert_to_langgraph(node)
        elif isinstance(node, AgentSpecOutputMessageNode):
            return self._output_message_node_convert_to_langgraph(node)
        elif isinstance(node, AgentSpecMapNode):
            return self._map_node_convert_to_langgraph(
                node,
                tool_registry=tool_registry,
                converted_components=converted_components,
                checkpointer=checkpointer,
                config=config,
                middleware=middleware,
            )
        else:
            raise NotImplementedError(
                f"The AgentSpec component of type {type(node)} is not yet supported for conversion"
            )

    def _input_message_node_convert_to_langgraph(
        self,
        node: AgentSpecInputMessageNode,
    ) -> "NodeExecutor":
        from pyagentspec.adapters.langgraph._node_execution import InputMessageNodeExecutor

        return InputMessageNodeExecutor(node)

    def _output_message_node_convert_to_langgraph(
        self,
        node: AgentSpecOutputMessageNode,
    ) -> "NodeExecutor":
        from pyagentspec.adapters.langgraph._node_execution import OutputMessageNodeExecutor

        return OutputMessageNodeExecutor(node)

    def _map_node_convert_to_langgraph(
        self,
        map_node: AgentSpecMapNode,
        tool_registry: Dict[str, "LangGraphTool"],
        converted_components: Dict[str, Any],
        checkpointer: Optional[Checkpointer],
        config: RunnableConfig,
        middleware: List[Any],
    ) -> "NodeExecutor":
        from pyagentspec.adapters.langgraph._node_execution import MapNodeExecutor

        subflow = self.convert(
            map_node.subflow,
            tool_registry=tool_registry,
            converted_components=converted_components,
            checkpointer=checkpointer,
            config=config,
            middleware=middleware,
        )
        if not isinstance(subflow, CompiledStateGraph):
            raise TypeError("MapNodeExecutor can only be initialized with MapNode")

        return MapNodeExecutor(map_node, subflow, config)

    def _flow_node_convert_to_langgraph(
        self,
        flow_node: AgentSpecFlowNode,
        tool_registry: Dict[str, "LangGraphTool"],
        converted_components: Dict[str, Any],
        checkpointer: Optional[Checkpointer],
        config: RunnableConfig,
        middleware: List[Any],
    ) -> "NodeExecutor":
        from pyagentspec.adapters.langgraph._node_execution import FlowNodeExecutor

        subflow = self.convert(
            flow_node.subflow,
            tool_registry=tool_registry,
            converted_components=converted_components,
            checkpointer=checkpointer,
            config=config,
            middleware=middleware,
        )
        if not isinstance(subflow, CompiledStateGraph):
            raise TypeError("FlowNodeExecutor can only initialize FlowNode")

        return FlowNodeExecutor(
            flow_node,
            subflow,
            config,
        )

    def _catch_exception_node_convert_to_langgraph(
        self,
        catch_node: AgentSpecCatchExceptionNode,
        tool_registry: Dict[str, "LangGraphTool"],
        converted_components: Dict[str, Any],
        checkpointer: Optional[Checkpointer],
        config: RunnableConfig,
        middleware: List[Any],
    ) -> "NodeExecutor":
        from pyagentspec.adapters.langgraph._node_execution import CatchExceptionNodeExecutor

        subflow = self.convert(
            catch_node.subflow,
            tool_registry=tool_registry,
            converted_components=converted_components,
            checkpointer=checkpointer,
            config=config,
            middleware=middleware,
        )
        if not isinstance(subflow, CompiledStateGraph):
            raise TypeError(
                "Internal error: CatchExceptionNodeExecutor expects `subflow` "
                f"to be a CompiledStateGraph, was {type(subflow)}"
            )

        return CatchExceptionNodeExecutor(
            catch_node,
            subflow,
            config,
        )

    def _api_node_convert_to_langgraph(self, api_node: AgentSpecApiNode) -> "NodeExecutor":
        from pyagentspec.adapters.langgraph._node_execution import ApiNodeExecutor

        return ApiNodeExecutor(api_node)

    def _branching_node_convert_to_langgraph(
        self, branching_node: AgentSpecBranchingNode
    ) -> "NodeExecutor":
        from pyagentspec.adapters.langgraph._node_execution import BranchingNodeExecutor

        return BranchingNodeExecutor(branching_node)

    def _agent_node_convert_to_langgraph(
        self,
        agent_node: AgentSpecAgentNode,
        tool_registry: Dict[str, "LangGraphTool"],
        converted_components: Dict[str, Any],
        checkpointer: Optional[Checkpointer],
        config: RunnableConfig,
        middleware: List[Any],
    ) -> "NodeExecutor":
        from pyagentspec.adapters.langgraph._node_execution import AgentNodeExecutor

        return AgentNodeExecutor(
            agent_node,
            tool_registry=tool_registry,
            converted_components=converted_components,
            checkpointer=checkpointer,
            config=config,
            middleware=middleware,
        )

    def _llm_node_convert_to_langgraph(
        self,
        llm_node: AgentSpecLlmNode,
        tool_registry: Dict[str, "LangGraphTool"],
        converted_components: Dict[str, Any],
        checkpointer: Optional[Checkpointer],
        config: RunnableConfig,
    ) -> "NodeExecutor":
        from pyagentspec.adapters.langgraph._node_execution import LlmNodeExecutor

        llm: BaseChatModel = self.convert(
            llm_node.llm_config,
            tool_registry=tool_registry,
            converted_components=converted_components,
            checkpointer=checkpointer,
            config=config,
        )
        return LlmNodeExecutor(llm_node, llm)

    def _tool_node_convert_to_langgraph(
        self,
        tool_node: AgentSpecToolNode,
        tool_registry: Dict[str, "LangGraphTool"],
        converted_components: Dict[str, Any],
        checkpointer: Optional[Checkpointer],
        config: RunnableConfig,
    ) -> "NodeExecutor":
        from pyagentspec.adapters.langgraph._node_execution import ToolNodeExecutor

        agentspec_tool = tool_node.tool
        tool_outputs = agentspec_tool.outputs or []

        # When a confirmation tool declares multiple outputs, a denial returns a single
        # string that cannot be mapped to those outputs. Bypass self.convert() for all
        # affected tool types so we can pass raise_on_denial=True: on rejection the
        # tool raises RuntimeError with a clear message instead of an opaque crash.
        if agentspec_tool.requires_confirmation and len(tool_outputs) > 1:
            _ensure_checkpointer_and_valid_tool_config(agentspec_tool, checkpointer)
            if isinstance(agentspec_tool, AgentSpecServerTool):
                tool = self._server_tool_convert_to_langgraph(
                    agentspec_tool, tool_registry, config=config, raise_on_denial=True
                )
            elif isinstance(agentspec_tool, AgentSpecRemoteTool):
                tool = self._remote_tool_convert_to_langgraph(
                    agentspec_tool, config=config, raise_on_denial=True
                )
            elif isinstance(agentspec_tool, AgentSpecClientTool):
                tool = self._client_tool_convert_to_langgraph(
                    agentspec_tool, raise_on_denial=True
                )
            else:
                raise ValueError(
                    f"Tool '{agentspec_tool.name}' of type "
                    f"'{type(agentspec_tool).__name__}' declares multiple outputs and "
                    f"requires_confirmation=True inside Flow ToolNode '{tool_node.name}'. "
                    f"Multi-output confirmation is supported for ServerTool, RemoteTool, "
                    f"and ClientTool. Use a single output or a supported tool type."
                )
        else:
            tool = self.convert(
                tool_node.tool,
                tool_registry=tool_registry,
                converted_components=converted_components,
                checkpointer=checkpointer,
                config=config,
            )

        return ToolNodeExecutor(tool_node, tool)

    def _end_node_convert_to_langgraph(self, end_node: AgentSpecEndNode) -> "NodeExecutor":
        from pyagentspec.adapters.langgraph._node_execution import EndNodeExecutor

        return EndNodeExecutor(end_node)

    def _start_node_convert_to_langgraph(self, start_node: AgentSpecStartNode) -> "NodeExecutor":
        from pyagentspec.adapters.langgraph._node_execution import StartNodeExecutor

        return StartNodeExecutor(start_node)

    def _remote_tool_convert_to_langgraph(
        self,
        remote_tool: AgentSpecRemoteTool,
        config: RunnableConfig,
        raise_on_denial: bool = False,
    ) -> StructuredTool:
        tool_name = remote_tool.name
        tool_description = remote_tool.description or ""
        _remote_tool = _confirm_then(
            func=_create_remote_tool_func(remote_tool),
            tool_name=tool_name,
            requires_confirmation=remote_tool.requires_confirmation,
            raise_on_denial=raise_on_denial,
        )

        # Use a Pydantic model for args_schema
        args_model = create_pydantic_model_from_properties(
            f"{tool_name}Args",
            remote_tool.inputs or [],
        )

        structured_tool = StructuredTool(
            name=tool_name,
            description=tool_description,
            args_schema=args_model,
            func=_remote_tool,
            coroutine=_as_structured_tool_coroutine(_remote_tool),
            callbacks=[
                AgentSpecToolCallbackHandler(tool=remote_tool),
            ],
        )
        return structured_tool

    def _server_tool_convert_to_langgraph(
        self,
        agentspec_server_tool: AgentSpecServerTool,
        tool_registry: Dict[str, LangGraphTool],
        config: RunnableConfig,
        raise_on_denial: bool = False,
    ) -> StructuredTool:
        def _is_structured_tool(x: Any) -> TypeGuard[StructuredTool]:
            return isinstance(x, StructuredTool)

        if agentspec_server_tool.name not in tool_registry:
            raise ValueError(
                f"The Agent Spec representation includes a tool '{agentspec_server_tool.name}' "
                f"but this tool does not appear in the tool registry"
            )

        tool_obj = tool_registry[agentspec_server_tool.name]
        tool_name = agentspec_server_tool.name
        tool_description = agentspec_server_tool.description or ""
        requires_confirmation = agentspec_server_tool.requires_confirmation
        structured_tool_name: str
        structured_tool_description: str
        args_schema: Union[type[BaseModel], Dict[str, Any]]
        if not (
            _is_structured_tool(tool_obj) or isinstance(tool_obj, BaseTool) or callable(tool_obj)
        ):
            raise TypeError(
                f"Unsupported tool type for '{tool_name}': {type(tool_obj)}. Expected callable, StructuredTool, or supported BaseTool."
            )

        tool_callable_kwargs = _get_structured_tool_callable_kwargs(
            tool_obj,
            tool_name=tool_name,
            requires_confirmation=requires_confirmation,
            raise_on_denial=raise_on_denial,
        )
        if _is_structured_tool(tool_obj):
            if not tool_callable_kwargs:
                raise TypeError(
                    f"Unsupported tool type for '{tool_name}': StructuredTool has neither func nor coroutine."
                )
            if tool_obj.args_schema is None:
                raise TypeError(
                    f"Unsupported tool type for '{tool_name}': StructuredTool has no args_schema."
                )

            structured_tool_name = tool_obj.name
            structured_tool_description = tool_obj.description
            args_schema = tool_obj.args_schema

        elif isinstance(tool_obj, BaseTool):
            if not tool_callable_kwargs:
                raise TypeError(
                    f"Unsupported tool type for '{tool_name}': BaseTool has neither func nor coroutine."
                )

            base_tool_args_schema = tool_obj.args_schema
            if base_tool_args_schema is None:
                base_tool_args_schema = create_pydantic_model_from_properties(
                    f"{tool_name}Args",
                    agentspec_server_tool.inputs or [],
                )
            structured_tool_name = tool_obj.name
            structured_tool_description = tool_obj.description
            args_schema = base_tool_args_schema
        else:
            structured_tool_name = tool_name
            structured_tool_description = tool_description
            args_schema = create_pydantic_model_from_properties(
                f"{tool_name}Args",
                agentspec_server_tool.inputs or [],
            )

        return StructuredTool(
            name=structured_tool_name,
            description=structured_tool_description,
            args_schema=args_schema,
            callbacks=[
                AgentSpecToolCallbackHandler(tool=agentspec_server_tool),
            ],
            **tool_callable_kwargs,
        )

    def _client_tool_convert_to_langgraph(
        self, agentspec_client_tool: AgentSpecClientTool, raise_on_denial: bool = False
    ) -> StructuredTool:
        # Warn at load time for Python < 3.11 since client tools use interrupt under the hood.
        if sys.version_info < (3, 11):
            logging.getLogger("pyagentspec.adapters.langgraph").warning(
                "Async interrupts on Python < 3.11 may raise 'Called get_config outside of a runnable context'. "
                "Prefer invoke/stream or upgrade to Python 3.11+ for ainvoke/astream."
            )

        tool_name = agentspec_client_tool.name
        tool_description = agentspec_client_tool.description or ""
        requires_confirmation = agentspec_client_tool.requires_confirmation

        def client_tool(*args: Any, **kwargs: Any) -> Any:
            if requires_confirmation:
                if args:
                    raise ValueError("Args are not supported, please only use kwargs")
                confirmed, reason = _confirm_tool_use(tool_name, **kwargs)

                if not confirmed:
                    if raise_on_denial:
                        raise RuntimeError(
                            f"Tool '{tool_name}' was denied by the user (reason: {reason}). "
                            f"Denial cannot be mapped to multiple declared outputs."
                        )
                    return f"Tool '{tool_name}' was denied execution by the user. Reason: {reason}"

            tool_request = {
                "type": "client_tool_request",
                "name": tool_name,
                "description": tool_description,
                "inputs": {
                    "args": args,
                    "kwargs": kwargs,
                },
            }
            response = interrupt(tool_request)
            return response

        # Use a Pydantic model for args_schema
        args_model = create_pydantic_model_from_properties(
            f"{tool_name}Args",
            agentspec_client_tool.inputs or [],
        )

        structured_tool = StructuredTool(
            name=tool_name,
            description=tool_description,
            args_schema=args_model,
            func=client_tool,
            coroutine=_as_structured_tool_coroutine(client_tool),
            # We do not add the tool execution callback here as it's not expected for client tools
        )
        return structured_tool

    def _mcp_tool_convert_to_langgraph(
        self,
        agentspec_mcp_tool: AgentSpecMCPTool,
        tool_registry: Dict[str, LangGraphTool],
        converted_components: Dict[str, Any],
    ) -> BaseTool:
        connection = self.convert(
            agentspec_mcp_tool.client_transport,
            tool_registry=tool_registry,
            converted_components=converted_components,
        )
        exposed_tools = self._get_or_create_langgraph_mcp_tools(
            client_transport=agentspec_mcp_tool.client_transport,
            langgraph_connection=connection,
            connection_key=agentspec_mcp_tool.client_transport.id,
            tool_registry=tool_registry,
        )
        return exposed_tools[agentspec_mcp_tool.name]

    def _mcp_toolbox_convert_to_langgraph(
        self,
        agentspec_mcp_toolbox: AgentSpecMCPToolBox,
        tool_registry: Dict[str, LangGraphTool],
        converted_components: Dict[str, Any],
    ) -> List[BaseTool]:
        connection = self.convert(
            agentspec_mcp_toolbox.client_transport,
            tool_registry=tool_registry,
            converted_components=converted_components,
        )
        remote_tools = self._get_or_create_langgraph_mcp_tools(
            client_transport=agentspec_mcp_toolbox.client_transport,
            langgraph_connection=connection,
            connection_key=agentspec_mcp_toolbox.client_transport.id,
            tool_registry=tool_registry,
        )
        # Below is logic to filter tools based on the tool_filter attribute of the toolbox
        # Normalize filter to {name: ToolSpec|None} (where None is when the filter is a string)
        filter_map = {
            (filter if isinstance(filter, str) else filter.name): (
                None if isinstance(filter, str) else filter
            )
            for filter in (agentspec_mcp_toolbox.tool_filter or [])
        }
        # If no filter provided, return all tools
        if not filter_map:
            filtered_tools = list(remote_tools.values())
        else:
            # Find missing by name first
            missing = sorted(name for name in filter_map if name not in remote_tools)
            if missing:
                raise ValueError("Missing tools: " + ", ".join(missing))
            # Validate specs (when provided) and collect tools
            for name, spec in filter_map.items():
                tool = remote_tools[name]
                if spec is not None and not _are_mcp_tool_spec_and_langchain_schemas_equal(
                    spec, tool
                ):
                    raise ValueError(
                        "Input descriptors mismatch for tool '%s'.\nLocal: %s\nRemote: %s"
                        % (spec.name, spec, getattr(tool, "args_schema", None))
                    )
            filtered_tools = [remote_tools[name] for name in filter_map]
        return filtered_tools

    def _swarm_convert_to_langgraph(
        self,
        agentspec_component: AgentSpecSwarm,
        tool_registry: Dict[str, LangGraphTool],
        converted_components: Dict[str, Any],
        checkpointer: Optional[Checkpointer],
        config: RunnableConfig,
        middleware: List[Any],
    ) -> CompiledStateGraph[Any, Any, Any]:
        if agentspec_component.handoff is AgentSpecHandoffMode.NEVER:
            # As of now, we cannot control what langgraph-swarm does internally in terms of conversation sharing.
            # The closest behaviors are OPTIONAL (probably best) or ALWAYS, but NEVER is not really supported.
            raise ValueError(
                "Handoff mode NEVER is not supported for conversion in LangGraph adapter"
            )
        members: Dict[str, AgentSpecComponent] = {
            # LangGraph distinguishes members by name, so we key by name.
            member.name: member
            # Relationships are tuples of (from_member, to_member)
            for member in (e for r in agentspec_component.relationships for e in r)
        }
        for member in members.values():
            # Swarm handoff is driven by an LLM emitting a transfer tool-call,
            # so a member must run a chat loop the handoff tools can attach to:
            # a plain Agent, or a ManagerWorkers (whose group-manager Agent
            # makes the decision). A Flow / nested Swarm has no single
            # "decide-the-handoff" LLM, so it cannot participate yet.
            if not isinstance(member, (AgentSpecAgent, AgentSpecManagerWorkers)):
                raise NotImplementedError(
                    "Only Agent and ManagerWorkers members are supported as part "
                    f"of a Swarm in the LangGraph adapter, received "
                    f"{type(member).__name__} instead."
                )
            # We convert each member even though we do not use this copy in the
            # swarm (the handoff-enabled copy is built below), so it is created
            # and stored in the converted-components registry in case it is
            # referenced elsewhere in the spec.
            self.convert(
                member,
                tool_registry=tool_registry,
                converted_components=converted_components,
                checkpointer=checkpointer,
                config=config,
                middleware=middleware,
            )
        handoffs: Dict[str, List[str]] = {name: [] for name in members}
        for from_member, to_member in agentspec_component.relationships:
            handoffs[from_member.name].append(to_member.name)
        # We re-create each member equipped with the handoff tools that let it
        # transfer the conversation to its related members.
        langgraph_members: List[CompiledStateGraph[Any, Any, Any]] = []
        for member in members.values():
            destinations = handoffs.get(member.name, [])
            if isinstance(member, AgentSpecManagerWorkers):
                # A ManagerWorkers member gets its handoff tools wired onto its
                # group-manager and re-emitted to the swarm — see
                # `_manager_workers_convert_to_langgraph`.
                langgraph_members.append(
                    self._manager_workers_convert_to_langgraph(
                        member,
                        tool_registry=tool_registry,
                        converted_components=converted_components,
                        checkpointer=checkpointer,
                        config=config,
                        middleware=middleware,
                        swarm_handoff_destinations=destinations,
                    )
                )
                continue
            agent = cast(AgentSpecAgent, member)
            langgraph_members.append(
                self._create_react_agent_with_given_info(
                    agent=agent,
                    name=agent.name,
                    system_prompt=agent.system_prompt,
                    llm_config=agent.llm_config,
                    tools=agent.tools,
                    toolboxes=agent.toolboxes,
                    inputs=agent.inputs or [],
                    outputs=agent.outputs or [],
                    tool_registry=tool_registry,
                    converted_components=converted_components,
                    checkpointer=checkpointer,
                    config=config,
                    middleware=middleware,
                    additional_langgraph_tools=[
                        langgraph_swarm.create_handoff_tool(agent_name=to_name)
                        for to_name in destinations
                    ],
                )
            )
        return langgraph_swarm.create_swarm(
            agents=langgraph_members,  # type: ignore
            default_active_agent=agentspec_component.first_agent.name,
        ).compile(name=agentspec_component.name, checkpointer=checkpointer)

    def _manager_workers_convert_to_langgraph(
        self,
        mw: AgentSpecManagerWorkers,
        tool_registry: Dict[str, "LangGraphTool"],
        converted_components: Dict[str, Any],
        checkpointer: Optional[Checkpointer],
        config: RunnableConfig,
        middleware: List[Any],
        swarm_handoff_destinations: Optional[List[str]] = None,
    ) -> CompiledStateGraph[Any, Any, Any]:
        """Compile a ``ManagerWorkers`` into a hierarchical LangGraph.

        Topology::

                          ┌─ delegate_to_w1 ─→ worker_1 ─┐
            START → manager ┤                              ├→ manager (loop)
                          └─ delegate_to_w2 ─→ worker_2 ─┘
                                  │
                                  └─ no tool_call ─→ END

        Each worker is recursively converted into a ``CompiledStateGraph``
        and wired in as a *subgraph node*, so ``astream_events`` exposes
        the parent/child boundary (``subgraph=True``) for tracing and SSE
        streaming. The manager is a react-agent given one synthetic
        ``delegate_to_<worker>`` tool per worker; the parent graph's
        conditional edge inspects the manager's last AIMessage to choose
        the next node, then the worker node runs in an isolated message
        context and emits a ``ToolMessage`` matched to the pending
        delegation tool-call id. Recursive ``ManagerWorkers`` (workers
        that are themselves ``ManagerWorkers``) compose for free through
        ``self.convert(...)``.

        ``swarm_handoff_destinations`` is set only when this ManagerWorkers is
        a member of a ``Swarm``: it lists the sibling member names this MW may
        hand the conversation off to. Each becomes a synthetic
        ``transfer_to_<sibling>`` tool on the manager, plus a ``__handoff__``
        parent node that re-emits the handoff up to the Swarm (the manager's
        own ``Command(graph=PARENT)`` would stop one level short, at this
        ManagerWorkers graph). When ``None``/empty (the standalone, Flow-step,
        or nested-worker case) no handoff machinery is added and behaviour is
        unchanged.
        """
        if not isinstance(mw.group_manager, AgentSpecAgent):
            # Pyagentspec allows any AgenticComponent as group_manager,
            # but the manager has to *decide* which worker to delegate to,
            # which means it needs a chat-LLM that emits tool_calls. Today
            # only Agent (and SpecializedAgent, a subclass) does that — a
            # Flow / Swarm / nested ManagerWorkers as the group_manager
            # doesn't have a "tool-call to delegate" output shape we can
            # route on.
            raise NotImplementedError(
                f"ManagerWorkers.group_manager must be an Agent for LangGraph "
                f"conversion; got {type(mw.group_manager).__name__}."
            )

        worker_node_names: List[str] = [
            _safe_node_name(worker.name, fallback_id=worker.id)
            for worker in mw.workers
        ]
        if len(set(worker_node_names)) != len(worker_node_names):
            raise ValueError(
                "ManagerWorkers worker names collide after normalization: "
                f"{worker_node_names}. Give each worker a unique name."
            )

        # 1. Recursively compile each worker as its own CompiledStateGraph.
        worker_graphs: Dict[str, CompiledStateGraph[Any, Any, Any]] = {}
        for worker, node_name in zip(mw.workers, worker_node_names):
            worker_graphs[node_name] = self.convert(
                worker,
                tool_registry=tool_registry,
                converted_components=converted_components,
                checkpointer=checkpointer,
                config=config,
                middleware=middleware,
            )

        # 2. Render the workers roster into the manager's system prompt
        #    so the LLM knows which delegation tool maps to which worker.
        manager_agent = mw.group_manager
        rendered_prompt = _append_workers_roster(
            manager_agent.system_prompt,
            [
                (node_name, worker.description or "")
                for worker, node_name in zip(mw.workers, worker_node_names)
            ],
        )

        # 3. Synthesize one delegation tool per worker. The tool body is a
        #    placeholder — the parent graph intercepts the manager's tool
        #    call before it executes and routes to the worker node.
        delegation_tools: List[Any] = [
            _make_worker_delegation_tool(node_name)
            for node_name in worker_node_names
        ]

        # 3b. When this ManagerWorkers is a Swarm member, synthesize one
        #     transfer_to_<sibling> tool per allowed handoff destination. Like
        #     the delegation tools these are placeholders: the manager's LLM
        #     emits the call, the parent graph intercepts it (routing edge
        #     below) and the `__handoff__` node re-emits the handoff to the
        #     Swarm. We map the (normalized) tool name back to the raw sibling
        #     name, which is the Swarm node name used as the handoff `goto`.
        handoff_dest_by_tool_name: Dict[str, str] = {}
        handoff_tools: List[Any] = []
        for dest in swarm_handoff_destinations or []:
            handoff_tool = _make_swarm_handoff_tool(dest)
            handoff_tools.append(handoff_tool)
            handoff_dest_by_tool_name[handoff_tool.name] = dest

        # 4. Compile the manager as a react-agent with the delegation tools
        #    (and, for a Swarm member, the transfer tools).
        manager_graph = self._create_react_agent_with_given_info(
            name=manager_agent.name,
            system_prompt=rendered_prompt,
            agent=manager_agent,
            llm_config=manager_agent.llm_config,
            tools=manager_agent.tools,
            toolboxes=manager_agent.toolboxes,
            inputs=manager_agent.inputs or [],
            outputs=manager_agent.outputs or [],
            tool_registry=tool_registry,
            converted_components=converted_components,
            checkpointer=checkpointer,
            config=config,
            middleware=middleware,
            additional_langgraph_tools=delegation_tools + handoff_tools,
        )

        # 5. Compose the parent StateGraph. The manager and every worker
        #    are CompiledStateGraphs added as subgraph nodes; LangGraph's
        #    streaming surfaces them with ``subgraph=True``.
        from langgraph.graph import MessagesState  # local: optional dep

        manager_node_key = _MANAGER_NODE_KEY
        if manager_node_key in worker_graphs:
            raise ValueError(
                f"Worker name '{manager_node_key}' is reserved for the "
                f"manager node in ManagerWorkers; rename the worker."
            )

        builder = StateGraph(MessagesState)
        builder.add_node(manager_node_key, manager_graph)
        for node_name, worker_graph in worker_graphs.items():
            builder.add_node(
                node_name,
                _wrap_worker_for_subgraph(worker_graph, node_name),
            )

        # Path-map covers delegate-to-worker and the END branch so langgraph
        # can statically validate the routing; the Swarm-handoff branch is
        # added only when this MW is a swarm member with handoff destinations.
        routing_path_map: Dict[str, str] = {
            node_name: node_name for node_name in worker_node_names
        }
        routing_path_map[langgraph_graph.END] = langgraph_graph.END
        if handoff_dest_by_tool_name:
            if _HANDOFF_NODE_KEY in worker_graphs:
                raise ValueError(
                    f"Worker name '{_HANDOFF_NODE_KEY}' is reserved for the "
                    f"Swarm-handoff node in ManagerWorkers; rename the worker."
                )
            builder.add_node(
                _HANDOFF_NODE_KEY,
                _make_handoff_forward_node(handoff_dest_by_tool_name),
            )
            routing_path_map[_HANDOFF_NODE_KEY] = _HANDOFF_NODE_KEY

        builder.add_edge(langgraph_graph.START, manager_node_key)
        builder.add_conditional_edges(
            manager_node_key,
            _route_manager_to_worker_handoff_or_end,
            routing_path_map,
        )
        for node_name in worker_node_names:
            builder.add_edge(node_name, manager_node_key)
        # The `__handoff__` node has no outgoing edge on purpose: it returns a
        # Command(goto=<sibling>, graph=PARENT) that exits this graph into the
        # Swarm, so looping it back to the manager would be wrong.

        compiled_graph = builder.compile(
            checkpointer=checkpointer, name=mw.name
        )

        # 6. Tracing — wrap stream/astream so ManagerWorkersExecutionSpan
        #    surrounds each run. Mirrors the patches applied to Agent and
        #    Flow graphs above.
        _patch_with_manager_workers_execution_span(compiled_graph, mw)
        return compiled_graph

    def _create_react_agent_with_given_info(
        self,
        *,
        name: str,
        system_prompt: str,
        agent: AgentSpecAgent,
        llm_config: AgentSpecLlmConfig,
        tools: List[AgentSpecTool],
        toolboxes: List[AgentSpecToolBox],
        inputs: List[AgentSpecProperty],
        outputs: List[AgentSpecProperty],
        tool_registry: Dict[str, LangGraphTool],
        converted_components: Dict[str, Any],
        checkpointer: Optional[Checkpointer],
        config: RunnableConfig,
        middleware: List[Any],
        additional_langgraph_tools: Optional[List[LangGraphTool]] = None,
    ) -> CompiledStateGraph[Any, Any, Any]:
        model = self.convert(
            llm_config,
            tool_registry=tool_registry,
            converted_components=converted_components,
            checkpointer=checkpointer,
            config=config,
        )
        langgraph_tools = (
            (additional_langgraph_tools or [])
            + [
                self.convert(
                    t,
                    tool_registry=tool_registry,
                    converted_components=converted_components,
                    checkpointer=checkpointer,
                    config=config,
                )
                for t in tools
            ]
            + [
                t
                for tb in toolboxes
                for t in self.convert(
                    tb,
                    tool_registry=tool_registry,
                    converted_components=converted_components,
                    checkpointer=checkpointer,
                    config=config,
                )
            ]
        )
        output_model: Optional[type[BaseModel]] = None
        state_schema: Optional[Any] = None

        # Build response (output) model (used for response_format). A single
        # string output is taken from the agent's final message (see
        # extract_outputs_from_invoke_result), so it needs no structured
        # generation — mirrors LlmNodeExecutor and lets a string output work on
        # models without structured-output support.
        if outputs and not is_single_string_output(outputs):
            output_model = create_pydantic_model_from_properties("AgentOutputModel", outputs)

        if inputs:
            state_schema = _create_agent_state_typed_dict(
                "AgentState",
                inputs=inputs,
            )

        create_agent_kwargs: Dict[str, Any] = dict(
            name=name,
            model=model,
            tools=langgraph_tools,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
            response_format=output_model,
            state_schema=state_schema,
        )
        if middleware:
            create_agent_kwargs["middleware"] = middleware
        compiled_graph = langchain_agents.create_agent(**create_agent_kwargs)

        # To enable flow execution traces monkey patch all the functions that invoke the compiled graph

        original_stream = compiled_graph.stream

        def patch_with_agent_execution_span(*args: Any, **kwargs: Any) -> Generator[Any, Any, Any]:
            span_name = f"AgentExecution[{agent.name}]"
            inputs = kwargs.get("input", {})
            if not isinstance(inputs, dict):
                inputs = {}
            with AgentSpecAgentExecutionSpan(name=span_name, agent=agent) as span:
                span.add_event(AgentSpecAgentExecutionStart(agent=agent, inputs=inputs))
                original_result: dict[str, Any] | Any = {}
                result: dict[str, Any]
                # This is going to patch stream and astream, that return iterators and yield chunks
                for chunk in original_stream(*args, **kwargs):
                    yield chunk
                    if isinstance(chunk, tuple):
                        original_result = chunk[1]
                if not isinstance(original_result, dict):
                    result = {}
                else:
                    result = original_result
                outputs = extract_outputs_from_invoke_result(result, agent.outputs or [])
                span.add_event(AgentSpecAgentExecutionEnd(agent=agent, outputs=outputs))

        original_astream = compiled_graph.astream

        async def patch_async_with_agent_execution_span(
            *args: Any, **kwargs: Any
        ) -> AsyncGenerator[Any, Any]:
            span_name = f"AgentExecution[{agent.name}]"
            inputs = kwargs.get("input", {})
            if not isinstance(inputs, dict):
                inputs = {}
            span = AgentSpecAgentExecutionSpan(name=span_name, agent=agent)
            try:
                await span.start_async()
            except NotImplementedError:
                span.start()
            try:
                try:
                    await span.add_event_async(
                        AgentSpecAgentExecutionStart(agent=agent, inputs=inputs)
                    )
                except NotImplementedError:
                    span.add_event(AgentSpecAgentExecutionStart(agent=agent, inputs=inputs))
                original_result: dict[str, Any] | Any = {}
                result: dict[str, Any]
                # This is going to patch stream and astream, that return iterators and yield chunks
                async for chunk in original_astream(*args, **kwargs):
                    yield chunk
                    if isinstance(chunk, tuple):
                        original_result = chunk[1]
                if not isinstance(original_result, dict):
                    result = {}
                else:
                    result = original_result

                outputs = extract_outputs_from_invoke_result(result, agent.outputs or [])
                try:
                    await span.add_event_async(
                        AgentSpecAgentExecutionEnd(agent=agent, outputs=outputs)
                    )
                except NotImplementedError:
                    span.add_event(AgentSpecAgentExecutionEnd(agent=agent, outputs=outputs))
            finally:
                try:
                    await span.end_async()
                except NotImplementedError:
                    span.end()

        # Monkey patch invocation functions to inject tracing
        # No need to patch `(a)invoke` as they internally use `(a)stream`
        compiled_graph.stream = patch_with_agent_execution_span  # type: ignore
        compiled_graph.astream = patch_async_with_agent_execution_span  # type: ignore
        return compiled_graph

    def _agent_convert_to_langgraph(
        self,
        agentspec_component: AgentSpecAgent,
        tool_registry: Dict[str, LangGraphTool],
        converted_components: Dict[str, Any],
        checkpointer: Optional[Checkpointer],
        config: RunnableConfig,
        middleware: List[Any],
    ) -> CompiledStateGraph[Any, Any, Any]:
        return self._create_react_agent_with_given_info(
            name=agentspec_component.name,
            system_prompt=agentspec_component.system_prompt,
            agent=agentspec_component,
            llm_config=agentspec_component.llm_config,
            tools=agentspec_component.tools,
            toolboxes=agentspec_component.toolboxes,
            inputs=agentspec_component.inputs or [],
            outputs=agentspec_component.outputs or [],
            tool_registry=tool_registry,
            converted_components=converted_components,
            checkpointer=checkpointer,
            config=config,
            middleware=middleware,
        )

    def _llm_convert_to_langgraph(
        self, llm_config: AgentSpecLlmConfig, config: RunnableConfig
    ) -> BaseChatModel:
        """Create the LLM model object for the chosen llm configuration."""
        generation_config = _generation_config_from_agentspec(
            llm_config.default_generation_parameters
        )

        use_responses_api = False
        if isinstance(llm_config, (OpenAiCompatibleConfig, OpenAiConfig)):
            use_responses_api = llm_config.api_type == OpenAIAPIType.RESPONSES

        callbacks: List[BaseCallbackHandler] = [
            AgentSpecLlmCallbackHandler(llm_config=llm_config),
        ]

        if isinstance(llm_config, VllmConfig):
            # if llm_config.api_key is None, ChatOpenAI constructor will attempt to read from the env
            # OPENAI_API_KEY and raise an error if missing
            # as local vLLM servers are not typically set up with API keys, we use the "EMPTY" as the default
            # for ease of use
            return _create_chat_openai_model(
                model_id=llm_config.model_id,
                base_url=_prepare_openai_compatible_url(llm_config.url),
                api_key=llm_config.api_key if llm_config.api_key is not None else "EMPTY",
                use_responses_api=use_responses_api,
                callbacks=callbacks,
                generation_config=generation_config,
                retry_config=self._retry_policy_convert_to_langgraph(llm_config.retry_policy),
            )
        elif isinstance(llm_config, OllamaConfig):
            if llm_config.retry_policy is not None:
                raise NotImplementedError(
                    "LangGraph ChatOllama conversion does not support `RetryPolicy`."
                )

            from langchain_ollama import ChatOllama

            return ChatOllama(
                base_url=llm_config.url,
                model=llm_config.model_id,
                callbacks=callbacks,
                temperature=generation_config.get("temperature"),
                num_predict=generation_config.get("max_tokens"),
                top_p=generation_config.get("top_p"),
            )
        elif isinstance(llm_config, OpenAiConfig):
            return _create_chat_openai_model(
                model_id=llm_config.model_id,
                api_key=llm_config.api_key,
                use_responses_api=use_responses_api,
                callbacks=callbacks,
                generation_config=generation_config,
                retry_config=self._retry_policy_convert_to_langgraph(llm_config.retry_policy),
            )
        elif isinstance(llm_config, OpenAiCompatibleConfig):
            return _create_chat_openai_model(
                model_id=llm_config.model_id,
                base_url=_prepare_openai_compatible_url(llm_config.url),
                api_key=llm_config.api_key,
                use_responses_api=use_responses_api,
                callbacks=callbacks,
                generation_config=generation_config,
                retry_config=self._retry_policy_convert_to_langgraph(llm_config.retry_policy),
            )
        elif isinstance(llm_config, OciGenAiConfig):
            if use_responses_api:
                raise NotImplementedError(
                    "OCI GenAI models with OpenAI Responses API is not yet supported"
                )

            if llm_config.retry_policy is not None:
                raise NotImplementedError(
                    "LangGraph OCI GenAI conversion does not support `RetryPolicy`."
                )

            from langchain_oci import ChatOCIGenAI  # type: ignore

            oci_model_kwargs: dict[str, int | float] = {}
            if "temperature" in generation_config:
                oci_model_kwargs["temperature"] = generation_config["temperature"]
            if "top_p" in generation_config:
                oci_model_kwargs["top_p"] = generation_config["top_p"]
            max_tokens = generation_config.get("max_tokens")
            if max_tokens is not None:
                token_key = (
                    "max_completion_tokens" if "openai" in llm_config.model_id else "max_tokens"
                )
                oci_model_kwargs[token_key] = max_tokens

            return ChatOCIGenAI(  # type: ignore
                model_id=llm_config.model_id,
                compartment_id=llm_config.compartment_id,
                model_kwargs=oci_model_kwargs,
                **self._oci_client_config_to_langgraph(llm_config.client_config),
            )
        else:
            # Bare LlmConfig — dispatch on api_provider string
            if llm_config.api_provider == "openai":
                return _create_chat_openai_model(
                    model_id=llm_config.model_id,
                    base_url=(
                        _ensure_url_has_scheme(llm_config.url)
                        if llm_config.url is not None
                        else None
                    ),
                    api_key=llm_config.api_key,
                    use_responses_api=llm_config.api_type == "responses",
                    callbacks=callbacks,
                    generation_config=generation_config,
                    retry_config=self._retry_policy_convert_to_langgraph(llm_config.retry_policy),
                )
            raise NotImplementedError(
                f"LlmConfig with api_provider='{llm_config.api_provider}' is not yet supported "
                f"in langgraph. Consider using a specific LlmConfig subclass instead."
            )

    def _retry_policy_convert_to_langgraph(
        self, retry_policy: Optional[RetryPolicy]
    ) -> "_ChatRetryConfig":
        """Convert Agent Spec retry policy settings into ChatOpenAI keyword arguments."""
        if retry_policy is None:
            return {}

        default_retry_policy = type(retry_policy)()
        unsupported_fields = [
            field_name
            for field_name in (
                "initial_retry_delay",
                "max_retry_delay",
                "backoff_factor",
                "jitter",
                "service_error_retry_on_any_5xx",
                "recoverable_statuses",
            )
            if getattr(retry_policy, field_name) != getattr(default_retry_policy, field_name)
        ]
        if unsupported_fields:
            raise NotImplementedError(
                "LangGraph ChatOpenAI conversion supports only "
                "`RetryPolicy.max_attempts` and `RetryPolicy.request_timeout`. "
                "This is because the underlying ChatOpenAI/OpenAI client only exposes "
                "retry count and timeout settings. "
                "Unsupported retry policy fields: " + ", ".join(unsupported_fields)
            )

        retry_config: _ChatRetryConfig = {"max_retries": retry_policy.max_attempts}
        if retry_policy.request_timeout is not None:
            retry_config["timeout"] = retry_policy.request_timeout
        return retry_config

    def _client_transport_convert_to_langgraph(
        self, agentspec_component: AgentSpecClientTransport
    ) -> "Union[StdioConnection, SSEConnection, StreamableHttpConnection]":
        import datetime

        from langchain_mcp_adapters.sessions import (
            SSEConnection,
            StdioConnection,
            StreamableHttpConnection,
        )

        sesh = agentspec_component.session_parameters.model_dump()
        sesh["read_timeout_seconds"] = datetime.timedelta(seconds=sesh["read_timeout_seconds"])
        if isinstance(agentspec_component, AgentSpecStdioTransport):
            return StdioConnection(
                transport="stdio",
                command=agentspec_component.command,
                args=agentspec_component.args,
                env=agentspec_component.env,
                cwd=agentspec_component.cwd,
                session_kwargs=sesh,
            )
        if isinstance(agentspec_component, AgentSpecSSEmTLSTransport):
            return SSEConnection(
                transport="sse",
                url=agentspec_component.url,
                headers=_remote_transport_headers(agentspec_component),
                httpx_client_factory=_HttpxClientFactory(
                    key_file=agentspec_component.key_file,
                    cert_file=agentspec_component.cert_file,
                    ssl_ca_cert=agentspec_component.ca_file,
                ),
            )
        if isinstance(agentspec_component, AgentSpecSSETransport):
            return SSEConnection(
                transport="sse",
                url=agentspec_component.url,
                headers=_remote_transport_headers(agentspec_component),
                httpx_client_factory=_HttpxClientFactory(verify=True),
            )
        if isinstance(agentspec_component, AgentSpecStreamableHTTPmTLSTransport):
            return StreamableHttpConnection(
                transport="streamable_http",
                url=agentspec_component.url,
                headers=_remote_transport_headers(agentspec_component),
                httpx_client_factory=_HttpxClientFactory(
                    key_file=agentspec_component.key_file,
                    cert_file=agentspec_component.cert_file,
                    ssl_ca_cert=agentspec_component.ca_file,
                ),
            )
        if isinstance(agentspec_component, AgentSpecStreamableHTTPTransport):
            return StreamableHttpConnection(
                transport="streamable_http",
                url=agentspec_component.url,
                headers=_remote_transport_headers(agentspec_component),
                httpx_client_factory=_HttpxClientFactory(verify=True),
            )
        raise ValueError(
            f"Agent Spec ClientTransport '{agentspec_component.__class__.__name__}' is not supported yet."
        )

    def _get_or_create_langgraph_mcp_tools(
        self,
        client_transport: AgentSpecClientTransport,
        langgraph_connection: "Union[StdioConnection, SSEConnection, StreamableHttpConnection]",
        connection_key: str,
        tool_registry: Dict[str, LangGraphTool],
    ) -> Dict[str, BaseTool]:
        """
        Synchronously load MCP tools and cache them into tool_registry keyed by:
        f"{agentspec_client_transport.id}::{tool_name}"

        Caching behavior:
        - If any tools are already present in the registry for this connection, returns those
        without reloading.
        - Otherwise, loads tools and inserts them atomically into the registry.
        """
        from langchain_mcp_adapters.tools import load_mcp_tools

        conn_prefix = f"{connection_key}::"
        existing = _get_session_tools_from_tool_registry(tool_registry, conn_prefix)
        if existing:
            return existing

        async def load_all_mcp_tools() -> List[BaseTool]:
            # Note: langchain supports session-specific MCP tools but we don't support that
            return await load_mcp_tools(session=None, connection=langgraph_connection)

        tools = run_async_in_sync(load_all_mcp_tools, method_name="load_mcp_tools")
        # We add callbacks to the tool for proper tracing
        for tool in tools:
            # Since we might not have the tool definition (e.g., in toolboxes)
            # we create the tool on-the-fly
            agentspec_tool = AgentSpecMCPTool(
                name=tool.name,
                description=tool.description,
                client_transport=client_transport,
                inputs=[
                    AgentSpecProperty(
                        title=arg_name, json_schema=_strip_schema_titles(arg_json_schema)
                    )
                    for arg_name, arg_json_schema in tool.args.items()
                ],
                outputs=[AgentSpecStringProperty(title="tool_output")],
            )
            if not tool.callbacks:
                tool.callbacks = []
            if isinstance(tool.callbacks, BaseCallbackHandler):
                tool.callbacks = [tool.callbacks]
            tool.callbacks.append(AgentSpecToolCallbackHandler(tool=agentspec_tool))  # type: ignore

        _add_session_tools_to_registry(tool_registry, tools, conn_prefix)

        return _get_session_tools_from_tool_registry(tool_registry, conn_prefix)

    def _oci_client_config_to_langgraph(
        self, client_config: OciClientConfig
    ) -> "_OciClientConfigKwargs":
        if isinstance(client_config, OciClientConfigWithSecurityToken):
            return {
                "auth_type": client_config.auth_type,
                "service_endpoint": client_config.service_endpoint,
                "auth_profile": client_config.auth_profile,
                "auth_file_location": client_config.auth_file_location,
            }
        elif isinstance(client_config, OciClientConfigWithInstancePrincipal):
            return {
                "auth_type": client_config.auth_type,
                "service_endpoint": client_config.service_endpoint,
            }
        elif isinstance(client_config, OciClientConfigWithResourcePrincipal):
            return {
                "auth_type": client_config.auth_type,
                "service_endpoint": client_config.service_endpoint,
            }
        elif isinstance(client_config, OciClientConfigWithApiKey):
            return {
                "auth_type": client_config.auth_type,
                "service_endpoint": client_config.service_endpoint,
                "auth_profile": client_config.auth_profile,
                "auth_file_location": client_config.auth_file_location,
            }
        else:
            raise ValueError(
                f"Agent Spec OciClientConfig '{client_config.__class__.__name__}' is not supported yet."
            )


def _get_session_tools_from_tool_registry(
    tool_registry: Dict[str, LangGraphTool], conn_prefix: str
) -> Dict[str, BaseTool]:
    return {
        key.replace(conn_prefix, ""): cast(BaseTool, tool)
        for key, tool in tool_registry.items()
        if key.startswith(conn_prefix)
    }


def _add_session_tools_to_registry(
    tool_registry: Dict[str, LangGraphTool], tools: List[BaseTool], conn_prefix: str
) -> None:
    # Prepare a staged mapping so we can insert all-or-nothing
    staged: Dict[str, BaseTool] = {}
    for tool in tools:
        # Derive a name: prefer .name, fallback to __name__ for callables
        tool_name = getattr(tool, "name", None)
        if not tool_name and callable(tool):
            tool_name = getattr(tool, "__name__", None)
        if not tool_name:
            raise ValueError("Loaded a tool without a name attribute or __name__.")

        key = f"{conn_prefix}{tool_name}"
        # Do not overwrite an existing entry if present
        if key not in tool_registry:
            staged[key] = tool
        else:
            raise ValueError(
                "Trying to add the same tool twice; this might happen "
                "when the tool is declared as both a standalone MCPTool and part of a MCPToolBox"
            )

    # Commit staged entries
    tool_registry.update(staged)


def _generation_config_from_agentspec(
    generation_parameters: Optional[LlmGenerationConfig],
) -> "_GenerationConfig":
    generation_config: _GenerationConfig = {}
    if generation_parameters is None:
        return generation_config
    if generation_parameters.temperature is not None:
        generation_config["temperature"] = generation_parameters.temperature
    if generation_parameters.max_tokens is not None:
        generation_config["max_tokens"] = generation_parameters.max_tokens
    if generation_parameters.top_p is not None:
        generation_config["top_p"] = generation_parameters.top_p
    return generation_config


def _create_chat_openai_model(
    *,
    model_id: str,
    use_responses_api: bool,
    callbacks: List[BaseCallbackHandler],
    generation_config: "_GenerationConfig",
    retry_config: "_ChatRetryConfig",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BaseChatModel:
    """Create a ChatOpenAI model without overriding env-based defaults.

    Important: passing `api_key=None` disables LangChain's env-based default (`OPENAI_API_KEY`)
    and results in a model without a sync client. Only pass `api_key` when it is explicitly
    specified in the Agent Spec config.
    """
    from langchain_openai import ChatOpenAI

    optional_kwargs: _ChatOpenAIOptionalKwargs = {}
    max_retries = retry_config.get("max_retries")
    if max_retries is not None:
        optional_kwargs["max_retries"] = max_retries
    timeout = retry_config.get("timeout")
    if timeout is not None:
        optional_kwargs["timeout"] = timeout
    if base_url is not None:
        optional_kwargs["base_url"] = base_url
    if api_key is not None:
        optional_kwargs["api_key"] = SecretStr(api_key)

    return ChatOpenAI(
        model=model_id,
        use_responses_api=use_responses_api,
        callbacks=callbacks,
        temperature=generation_config.get("temperature"),
        max_completion_tokens=generation_config.get("max_tokens"),
        top_p=generation_config.get("top_p"),
        **optional_kwargs,
    )


class _ChatOpenAIOptionalKwargs(TypedDict):
    """Optional ChatOpenAI arguments that must be omitted when unset."""

    max_retries: NotRequired[int]
    timeout: NotRequired[float]
    base_url: NotRequired[str]
    api_key: NotRequired[SecretStr]


class _GenerationConfig(TypedDict):
    """Normalized Agent Spec generation settings supported by LangGraph adapters."""

    temperature: NotRequired[float]
    max_tokens: NotRequired[int]
    top_p: NotRequired[float]


class _ChatRetryConfig(TypedDict):
    """Keyword arguments for ChatOpenAI retry and timeout configuration."""

    max_retries: NotRequired[int]
    timeout: NotRequired[float]


class _OciClientConfigKwargs(TypedDict):
    """Keyword arguments for OCI GenAI client authentication configuration."""

    auth_type: Required[str]
    service_endpoint: Required[str]
    auth_profile: NotRequired[str]
    auth_file_location: NotRequired[str]


def _ensure_url_has_scheme(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    return url


def _prepare_openai_compatible_url(url: str) -> str:
    """
    Correctly formats a URL for an OpenAI-compatible server.

    This function is robust and handles multiple formats:
    - Ensures a scheme (http, https) is present, defaulting to 'http'.
    - Replaces any existing path with exactly '/v1'.

    Examples:
        - "localhost:8000"          -> "http://localhost:8000/v1"
        - "127.0.0.1:5000"          -> "http://127.0.0.1:5000/v1"
        - "https://api.example.com"   -> "https://api.example.com/v1"
        - "http://my-host/api/v2"   -> "http://my-host/v1"
    """
    from urllib.parse import urlparse, urlunparse

    url = _ensure_url_has_scheme(url)
    parsed_url = urlparse(url)
    # parsed_url is a namedtuple object, and it has the _replace method
    # this is actually a public facing method, check python documentation of namedtuple
    v1_url_parts = parsed_url._replace(path="/v1", params="", query="", fragment="")
    final_url = urlunparse(v1_url_parts)

    return str(final_url)


def _are_mcp_tool_spec_and_langchain_schemas_equal(
    mcp_spec: AgentSpecMCPToolSpec, langchain_schema: BaseTool
) -> bool:
    if not isinstance(langchain_schema.args_schema, dict):
        raise ValueError(
            f"Expected Langchain StructuredTool.args_schema to be a dict but got {type(langchain_schema.args_schema)}"
        )
    agentspec_json_schemas = {
        inp.json_schema["title"]: _normalize_title(inp.json_schema)
        for inp in (mcp_spec.inputs or [])
    }
    langchain_json_schemas = {
        k: _normalize_title(v) for k, v in langchain_schema.args_schema["properties"].items()
    }
    return json_schemas_have_same_type(agentspec_json_schemas, langchain_json_schemas)


def _normalize_title(d: Dict[str, Any]) -> Dict[str, Any]:
    """If `title`, then lowercase."""
    out = dict(d)  # shallow copy
    if isinstance(out.get("title"), str):
        out["title"] = out["title"].lower()
    return out


def _strip_schema_titles(json_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``json_schema`` with ``title`` annotations removed.

    MCP servers commonly derive tool schemas from OpenAPI documents whose
    nested schemas carry human-readable titles (e.g. ``"Rich Text"``).
    ``Property`` validates every title it can reach as an identifier and
    rejects the whole schema, killing the run. The on-the-fly ``MCPTool``
    built from these schemas only backs the tracing callback (the LLM-facing
    ``args_schema`` is untouched), and its port name is passed explicitly as
    ``title=``, so the schema's own titles are safe to drop.

    Only the positions ``Property``'s validator traverses are visited —
    ``items``, ``anyOf``, ``additionalProperties`` and ``properties`` values.
    A generic recursive strip would corrupt non-schema payloads such as
    ``default``/``examples`` values containing a ``title`` key.
    """
    out = {key: value for key, value in json_schema.items() if key != "title"}
    if isinstance(out.get("items"), dict):
        out["items"] = _strip_schema_titles(out["items"])
    if isinstance(out.get("anyOf"), list):
        out["anyOf"] = [
            _strip_schema_titles(inner) if isinstance(inner, dict) else inner
            for inner in out["anyOf"]
        ]
    if isinstance(out.get("additionalProperties"), dict):
        out["additionalProperties"] = _strip_schema_titles(out["additionalProperties"])
    if isinstance(out.get("properties"), dict):
        out["properties"] = {
            name: _strip_schema_titles(inner) if isinstance(inner, dict) else inner
            for name, inner in out["properties"].items()
        }
    return out


def _confirm_tool_use(tool_name: str, **tool_arguments: Any) -> Tuple[bool, str]:
    # aligned with https://docs.langchain.com/oss/python/langchain/human-in-the-loop#responding-to-interrupts
    ALLOWED_DECISIONS = ["approve", "reject"]
    confirmation_payload = {
        "action_requests": [
            {
                "name": tool_name,
                "arguments": tool_arguments,
                "description": f"Tool execution pending approval\n\nTool: {tool_name}\nArgs: {tool_arguments}",
            }
        ],
        "review_configs": [
            {
                "action_name": tool_name,
                "allowed_decisions": ALLOWED_DECISIONS,
                "description": (
                    'Please resume with {"decisions": [{"type": "approve"}]}  # or "reject" '
                    'with an optional "reason" for rejected tool calls.'
                ),
            }
        ],
    }
    response = interrupt(confirmation_payload)
    if not isinstance(response, dict) or "decisions" not in response:
        raise ValueError(
            f"Tool confirmation result for tool {tool_name} is not valid, should be a "
            f"dict with a 'decisions' key, was {response!r} of type {type(response)}."
        )
    decision_list = response["decisions"]
    if len(decision_list) != 1:
        raise ValueError(
            f"Tool confirmation result for tool {tool_name} is not valid, decisions "
            f"should be of length 1, was of length {len(decision_list)}"
        )
    decision = decision_list[0]
    if "type" not in decision or not decision["type"] in ALLOWED_DECISIONS:
        raise ValueError(
            f"Tool confirmation result for tool {tool_name} is not valid, "
            f"decision should be in {ALLOWED_DECISIONS}, was {decision}."
        )

    return (decision["type"] == "approve"), decision.get("reason", "No reason was provided.")


@overload
def _confirm_then(
    func: Callable[..., Awaitable[Any]],
    tool_name: str,
    requires_confirmation: bool,
    raise_on_denial: bool = ...,
) -> Callable[..., Awaitable[Any]]: ...


@overload
def _confirm_then(
    func: Callable[..., Any],
    tool_name: str,
    requires_confirmation: bool,
    raise_on_denial: bool = ...,
) -> Callable[..., Any]: ...


def _confirm_then(
    func: Callable[..., Any],
    tool_name: str,
    requires_confirmation: bool,
    raise_on_denial: bool = False,
) -> Callable[..., Any]:
    """Wrap a callable so that it first interrupts for confirmation (if required).

    When raise_on_denial is True, denial raises RuntimeError instead of returning a
    string. Use this inside a Flow ToolNode with multiple outputs where a denial
    string cannot be mapped to the declared output structure.
    """
    if not requires_confirmation:
        return func

    if _is_async_callable(func):

        async def _wrapped_async(*args: Any, **kwargs: Any) -> Any:
            confirmation_arguments = {"args": args, **kwargs} if args else kwargs
            confirmed, reason = _confirm_tool_use(tool_name, **confirmation_arguments)

            if not confirmed:
                if raise_on_denial:
                    raise RuntimeError(
                        f"Tool '{tool_name}' was denied by the user (reason: {reason}). "
                        f"Denial cannot be mapped to multiple declared outputs."
                    )
                return f"Tool '{tool_name}' was denied execution by the user. Reason: {reason}"

            return await func(*args, **kwargs)

        return _wrapped_async

    def _wrapped_sync(*args: Any, **kwargs: Any) -> Any:
        confirmation_arguments = {"args": args, **kwargs} if args else kwargs
        confirmed, reason = _confirm_tool_use(tool_name, **confirmation_arguments)

        if not confirmed:
            if raise_on_denial:
                raise RuntimeError(
                    f"Tool '{tool_name}' was denied by the user (reason: {reason}). "
                    f"Denial cannot be mapped to multiple declared outputs."
                )
            return f"Tool '{tool_name}' was denied execution by the user. Reason: {reason}"

        return func(*args, **kwargs)

    return _wrapped_sync


def _is_async_callable(func: Callable[..., Any]) -> TypeGuard[Callable[..., Awaitable[Any]]]:
    return inspect.iscoroutinefunction(func) or inspect.iscoroutinefunction(
        getattr(func, "__call__", None)
    )


def _as_structured_tool_coroutine(
    func: Callable[..., Any],
) -> Callable[..., Awaitable[Any]]:
    if _is_async_callable(func):
        return func

    async def _wrapped_async(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapped_async


class StructuredToolCallableKwargs(TypedDict, total=False):
    func: Callable[..., Any]
    coroutine: Callable[..., Awaitable[Any]]


def _get_structured_tool_callable_kwargs(
    tool_obj: Union[StructuredTool, BaseTool, Callable[..., Any]],
    tool_name: str,
    requires_confirmation: bool = False,
    raise_on_denial: bool = False,
) -> StructuredToolCallableKwargs:
    """Return the callables to pass to StructuredTool.

    Bare callables map to exactly one field:
    - sync callable -> func
    - async callable -> coroutine

    Existing BaseTool instances may expose one or both fields already.
    We preserve those fields as-is and only wrap them for confirmation.
    """
    structured_tool_callable_kwargs: StructuredToolCallableKwargs = {}

    if isinstance(tool_obj, BaseTool):
        tool_func = getattr(tool_obj, "func", None)
        if tool_func is not None:
            structured_tool_callable_kwargs["func"] = _confirm_then(
                func=tool_func,
                tool_name=tool_name,
                requires_confirmation=requires_confirmation,
                raise_on_denial=raise_on_denial,
            )

        tool_coroutine = getattr(tool_obj, "coroutine", None)
        if tool_coroutine is not None:
            structured_tool_callable_kwargs["coroutine"] = _confirm_then(
                func=tool_coroutine,
                tool_name=tool_name,
                requires_confirmation=requires_confirmation,
                raise_on_denial=raise_on_denial,
            )
    elif callable(tool_obj):
        wrapped_tool = _confirm_then(
            func=tool_obj,
            tool_name=tool_name,
            requires_confirmation=requires_confirmation,
            raise_on_denial=raise_on_denial,
        )
        if _is_async_callable(wrapped_tool):
            structured_tool_callable_kwargs["coroutine"] = wrapped_tool
        else:
            structured_tool_callable_kwargs["func"] = wrapped_tool
    else:
        raise TypeError(
            f"Unsupported tool type for '{tool_name}': {type(tool_obj)}. Expected callable, StructuredTool, or supported BaseTool."
        )

    return structured_tool_callable_kwargs


def _ensure_checkpointer_and_valid_tool_config(
    agentspec_tool: AgentSpecTool, checkpointer: Optional[Checkpointer]
) -> None:
    tool_name = agentspec_tool.name
    if agentspec_tool.requires_confirmation and checkpointer is None:
        raise ValueError(
            f"A Checkpointer is required for tool '{tool_name}' because requires_confirmation=True"
        )
    elif isinstance(agentspec_tool, AgentSpecClientTool) and checkpointer is None:
        raise ValueError(f"A Checkpointer is required when using ClientTool '{tool_name}'.")


# ─── ManagerWorkers helpers ──────────────────────────────────────────────────

# Node key for the manager subgraph in the ManagerWorkers parent StateGraph.
# Chosen so it cannot collide with a normalized worker node name (which is
# always lowercase + [a-z0-9_]).
_MANAGER_NODE_KEY = "__manager__"

# Prefix the manager's LLM uses to address a delegation tool. The suffix is
# the normalized worker node name.
_DELEGATE_TOOL_PREFIX = "delegate_to_"

# Prefix the manager's LLM uses to hand the conversation off to a sibling
# Swarm member (only present when this ManagerWorkers is a Swarm member). The
# suffix is the normalized sibling name; matches langgraph_swarm's convention.
_HANDOFF_TOOL_PREFIX = "transfer_to_"

# Node key for the Swarm-handoff node in the ManagerWorkers parent StateGraph.
# Like ``_MANAGER_NODE_KEY`` it cannot collide with a normalized worker node
# name (which never contains leading/trailing underscores).
_HANDOFF_NODE_KEY = "__handoff__"

# Keys carried on the per-delegation ``Send`` payload from the manager's
# routing edge to a worker node, so a worker run knows which task it was
# given and which ``tool_call_id`` its reply ToolMessage must answer. This
# is what lets one manager turn delegate to several workers at once: each
# delegation routes as its own ``Send`` and is answered independently.
_DELEGATE_TASK_KEY = "__delegate_task__"
_DELEGATE_CALL_ID_KEY = "__delegate_tool_call_id__"

# Collapses any run of whitespace to a single space so multi-line worker
# descriptions stay on one roster line.
_WHITESPACE_RE = re.compile(r"\s+")


def _safe_node_name(name: str, fallback_id: str) -> str:
    """Normalize a worker name into a LangGraph node identifier.

    LangGraph node names must be hashable strings; in practice we want
    ASCII-friendly identifiers that also work as Python attribute-ish
    names (the LLM is going to see ``delegate_to_<node_name>`` as a tool
    name and needs to be able to emit it reliably). We lowercase, collapse
    non-alphanumerics to underscores, strip surrounding underscores, and
    fall back to the (component) id — normalized the same way — if the
    name yields an empty string. Falling through both transforms keeps
    node names internally consistent regardless of which input wins.
    """

    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", (s or "").lower()).strip("_")

    return _norm(name) or _norm(fallback_id) or "worker"


def _tc_get(tool_call: Any, key: str) -> Any:
    """Read ``key`` off a tool call that may be a dict or a pydantic-style
    object (langchain emits either depending on the message source)."""
    if isinstance(tool_call, dict):
        return tool_call.get(key)
    return getattr(tool_call, key, None)


def _append_workers_roster(
    system_prompt: str,
    entries: List[Tuple[str, str]],
) -> str:
    """Prepend the manager's system prompt with an ``Available workers:``
    roster block listing ``- <name>: <description>`` per worker.

    Each description has whitespace flattened so multi-line descriptions
    don't corrupt the one-line-per-worker block shape that the LLM relies
    on for routing.
    """
    if not entries:
        return system_prompt
    lines = [
        f"- {name}: {_WHITESPACE_RE.sub(' ', description).strip()}"
        for name, description in entries
    ]
    roster = "Available workers:\n" + "\n".join(lines)
    return f"{system_prompt}\n\n{roster}" if system_prompt else roster


def _make_worker_delegation_tool(worker_node_name: str) -> Any:
    """Build the ``delegate_to_<worker>`` tool the manager's LLM emits to
    route work to a worker subgraph.

    The tool body returns a ``Command(graph=Command.PARENT)`` which the
    langchain ``ToolNode`` propagates up to the parent graph, breaking out
    of the manager's react-agent inner loop. It deliberately carries **no
    ``goto``**: routing is the parent graph's conditional edge's job
    (:func:`_route_manager_to_worker_handoff_or_end`), which inspects the manager's
    AIMessage and fans out one ``Send`` per ``delegate_to_<worker>`` call.
    Routing via ``goto`` here would be wrong when the manager emits several
    delegations in a single turn — ``ToolNode`` collapses the multiple
    parent commands down to one, so only the first worker would run and the
    other delegations' ``tool_call_id``s would be left unanswered. Each
    worker run replies with a ToolMessage matched to its ``tool_call_id``;
    on the next manager turn the LLM sees every ``AIMessage(tool_call)`` +
    ``ToolMessage(worker_reply)`` pair and can produce its final answer — a
    well-formed tool-call / tool-result sequence in the OpenAI contract.

    Modelled on ``langgraph_swarm.create_handoff_tool``, which uses the
    same Command-propagation pattern for swarm handoffs.
    """
    from typing import Annotated

    from langchain_core.tools import InjectedToolCallId, tool
    from langgraph.prebuilt import InjectedState
    from langgraph.types import Command

    tool_name = f"{_DELEGATE_TOOL_PREFIX}{worker_node_name}"

    @tool(tool_name)
    def _delegate(
        task: str,
        state: Annotated[Any, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Delegate a task to the named worker and wait for its reply.

        ``task`` is the natural-language instruction the worker should
        execute. The worker runs in its own isolated message context;
        only this ``task`` is forwarded as the worker's first message.
        """
        # Mirror langgraph_swarm's create_handoff_tool: project the
        # subgraph's messages — including the AIMessage carrying this
        # tool_call — onto the PARENT state via the update. The
        # ``add_messages`` reducer dedupes by id so existing messages
        # aren't duplicated. The parent's routing edge then reads the
        # AIMessage off the parent state and fans out a worker run per
        # delegation, each answering its own ``tool_call_id``.
        subgraph_messages: List[Any] = []
        if isinstance(state, dict):
            subgraph_messages = list(state.get("messages") or [])
        else:
            subgraph_messages = list(getattr(state, "messages", []) or [])
        del task, tool_call_id  # task + id are recovered by the routing edge
        return Command(
            graph=Command.PARENT,
            update={"messages": subgraph_messages},
        )

    _delegate.description = (
        f"Delegate a task to the {worker_node_name} worker and receive "
        f"its response. Use this when the task fits the worker's "
        f"described capability."
    )
    return _delegate


def _handoff_tool_name(destination_name: str) -> str:
    """``transfer_to_<normalized sibling>`` — the tool name the manager's LLM
    emits to hand off to a Swarm sibling. Normalization mirrors
    :func:`_safe_node_name` so the name is a clean tool identifier; the raw
    destination (the Swarm node name used as the handoff ``goto``) is recovered
    via the tool-name→destination map held by the handoff node."""
    normalized = re.sub(r"[^a-z0-9]+", "_", (destination_name or "").lower()).strip("_")
    return f"{_HANDOFF_TOOL_PREFIX}{normalized or 'agent'}"


def _make_swarm_handoff_tool(destination_name: str) -> Any:
    """Build the ``transfer_to_<sibling>`` tool a Swarm-member ManagerWorkers'
    manager LLM emits to hand the conversation off to a sibling member.

    Mirrors :func:`_make_worker_delegation_tool`: the body returns a
    ``Command(graph=Command.PARENT)`` carrying **no** ``goto`` — it only breaks
    the manager's react loop and surfaces the AIMessage on the parent graph.
    The real handoff is performed by the parent's handoff node
    (:func:`_make_handoff_forward_node`), which re-emits a
    ``Command(goto=<sibling>, graph=Command.PARENT)`` so the destination is
    resolved against the *Swarm* graph. A plain
    ``langgraph_swarm.create_handoff_tool`` cannot be used directly on the
    manager: its single ``Command(graph=PARENT)`` would land on *this*
    ManagerWorkers graph (one level short of the Swarm) and be silently
    dropped, so the handoff would never happen.
    """
    from typing import Annotated

    from langchain_core.tools import InjectedToolCallId, tool
    from langgraph.prebuilt import InjectedState
    from langgraph.types import Command

    tool_name = _handoff_tool_name(destination_name)

    @tool(tool_name)
    def _handoff(
        state: Annotated[Any, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Hand the whole conversation off to the named agent."""
        if isinstance(state, dict):
            subgraph_messages = list(state.get("messages") or [])
        else:
            subgraph_messages = list(getattr(state, "messages", []) or [])
        del tool_call_id  # id is recovered from the AIMessage by the handoff node
        return Command(
            graph=Command.PARENT,
            update={"messages": subgraph_messages},
        )

    _handoff.description = (
        f"Transfer the full conversation to the '{destination_name}' agent so "
        f"it takes over the dialogue with the user. Use this when "
        f"'{destination_name}' is better suited to continue; you will not "
        f"regain control afterwards."
    )
    return _handoff


def _make_handoff_forward_node(handoff_dest_by_tool_name: Dict[str, str]) -> Any:
    """Build the parent-graph node that performs a Swarm handoff.

    Reached (via the routing edge) when the manager's last AIMessage carries a
    ``transfer_to_<sibling>`` tool call. It returns a
    ``Command(goto=<sibling>, graph=Command.PARENT, update={..., active_agent})``
    that re-emits the handoff up to the Swarm graph (mirroring what
    ``langgraph_swarm.create_handoff_tool`` does for a plain Agent member).

    Transcript validity: the ManagerWorkers graph exits via this PARENT
    command rather than running to its own END, so its internal messages do
    **not** merge into the shared Swarm conversation — only this command's
    ``update`` does. We therefore forward the manager's transfer AIMessage
    itself, followed by a ``ToolMessage`` answering every (still-unanswered)
    tool call on it, so the Swarm sees a well-formed
    ``AIMessage(tool_calls)`` → ``ToolMessage`` sequence — an orphan
    ToolMessage would 400 the next member's LLM. Answering *every* call (not
    just the transfer) keeps the sequence valid even when the manager emitted
    delegations in the same turn; the manager's internal delegation mechanics
    otherwise stay hidden inside the ManagerWorkers, as in a standalone run.

    Returns a ``RunnableLambda`` exposing both sync (``func``) and async
    (``afunc``) entrypoints so LangGraph can call it on either path; the body
    is pure so the async wrapper just delegates.
    """
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from pyagentspec.adapters.langgraph._types import RunnableLambda

    def _forward(state: Dict[str, Any]) -> Any:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        tool_calls = getattr(last, "tool_calls", None) or []
        transfer_call = next(
            (
                tc
                for tc in tool_calls
                if _tc_get(tc, "name") in handoff_dest_by_tool_name
            ),
            None,
        )
        if transfer_call is None:
            # Defensive: the routing edge only sends us here on a transfer call.
            return {"messages": []}
        destination = handoff_dest_by_tool_name[_tc_get(transfer_call, "name")]
        transfer_id = _tc_get(transfer_call, "id") or ""
        already_answered = {
            getattr(m, "tool_call_id", None)
            for m in messages
            if getattr(m, "type", None) == "tool"
        }
        tool_messages: List[Any] = []
        for tc in tool_calls:
            call_id = _tc_get(tc, "id") or ""
            if call_id in already_answered:
                continue
            if call_id == transfer_id:
                content = f"Successfully transferred to {destination}"
            else:
                content = (
                    f"Not executed: the conversation was handed off to "
                    f"{destination}."
                )
            tool_messages.append(
                ToolMessage(
                    content=content,
                    name=_tc_get(tc, "name"),
                    tool_call_id=call_id,
                )
            )
        return Command(
            goto=destination,
            graph=Command.PARENT,
            # Forward the manager's transfer AIMessage ahead of the answering
            # ToolMessages so the Swarm transcript is a valid tool-call/result
            # sequence (the MW's internal messages don't otherwise merge on a
            # PARENT-jump exit). add_messages dedupes by id, so re-sending an
            # already-present message is a no-op.
            update={"messages": [last, *tool_messages], "active_agent": destination},
        )

    async def _forward_async(state: Dict[str, Any]) -> Any:
        return _forward(state)

    return RunnableLambda(
        func=_forward,
        afunc=_forward_async,
        name="swarm_handoff",
    )


def _is_handoff_name(name: Any) -> bool:
    """True if ``name`` is one of the synthetic ``transfer_to_<sibling>`` tool
    names the manager emits to hand the conversation off to a Swarm sibling."""
    return isinstance(name, str) and name.startswith(_HANDOFF_TOOL_PREFIX)


def _route_manager_to_worker_handoff_or_end(state: Dict[str, Any]) -> Any:
    """Inspect the manager's last AIMessage and route the parent graph.

    Routing precedence:

    * a ``transfer_to_<sibling>`` tool call → the Swarm-handoff node
      (:func:`_make_handoff_forward_node`), which re-emits the handoff up to
      the Swarm. Handoff transfers the *whole* conversation, so it wins over
      delegation and routes to a single destination (only present when this
      ManagerWorkers is a Swarm member).
    * one or more ``delegate_to_<worker>`` tool calls → one ``Send`` per
      delegation; otherwise → ``END``.

    A single manager turn may delegate to several workers at once — the LLM
    emits multiple ``delegate_to_<worker>`` tool calls in one AIMessage
    (e.g. "spin up 5 sub-agents"). Every one of those tool calls must be
    answered by its own ``ToolMessage`` matched to the originating
    ``tool_call_id``; leaving any unanswered produces a tool-call /
    tool-result mismatch that violates the OpenAI contract and makes the
    manager hallucinate the missing replies. We therefore emit one ``Send``
    per delegation, each carrying the delegated ``task`` and its
    ``tool_call_id`` so the target worker node can reply to exactly that
    call. Multiple ``Send``s to the same worker node run as independent
    tasks. Plain (non-delegation) tool calls were already executed inside
    the manager's react-agent loop before routing reaches here.
    """
    from langgraph.types import Send

    messages = state.get("messages") or []
    if not messages:
        return langgraph_graph.END
    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    # Swarm handoff wins: it hands the whole conversation to a sibling, so any
    # delegations in the same turn are answered-and-dropped by the handoff node.
    if any(_is_handoff_name(_tc_get(tc, "name")) for tc in tool_calls):
        return _HANDOFF_NODE_KEY
    sends = []
    for tc in tool_calls:
        name = _tc_get(tc, "name")
        if isinstance(name, str) and name.startswith(_DELEGATE_TOOL_PREFIX):
            worker_node_name = name[len(_DELEGATE_TOOL_PREFIX):]
            args = _tc_get(tc, "args") or {}
            sends.append(
                Send(
                    worker_node_name,
                    {
                        _DELEGATE_TASK_KEY: args.get("task") or "",
                        _DELEGATE_CALL_ID_KEY: _tc_get(tc, "id") or "",
                    },
                )
            )
    return sends or langgraph_graph.END


def _wrap_worker_for_subgraph(
    worker_graph: CompiledStateGraph[Any, Any, Any],
    worker_node_name: str,
) -> Any:
    """Wrap a worker subgraph so it runs with an isolated ``messages``
    context (the delegation task only) and its final reply comes back as
    a ToolMessage matched to the manager's pending delegation tool-call.

    This is what makes a ManagerWorkers parent graph hierarchical rather
    than a shared-state Swarm: workers do NOT see each other's messages,
    and only one message — the manager's chosen task — is forwarded to
    each worker run. The worker's last AIMessage content is captured as
    the ToolMessage content so the manager's react-agent loop sees a
    well-formed tool response on the next turn.

    Returns a ``RunnableLambda`` exposing both sync (``func``) and async
    (``afunc``) entrypoints — LangGraph picks the right one based on
    whether the parent graph is invoked via ``invoke`` or ``ainvoke``.
    """
    from langchain_core.messages import SystemMessage, ToolMessage

    from pyagentspec.adapters.langgraph._types import RunnableLambda

    delegate_tool_name = f"{_DELEGATE_TOOL_PREFIX}{worker_node_name}"

    def _extract_pending(state: Dict[str, Any]) -> Tuple[str, str]:
        # Fan-out path: the routing edge's ``Send`` payload carries this
        # delegation's task and its originating tool_call_id directly, so a
        # single manager turn can delegate to this worker more than once
        # without the runs colliding on a shared "first pending call".
        if isinstance(state, dict) and _DELEGATE_CALL_ID_KEY in state:
            return (
                state.get(_DELEGATE_TASK_KEY) or "",
                state.get(_DELEGATE_CALL_ID_KEY) or "",
            )
        # Direct-edge path (a worker wired in without Send): recover task +
        # id from the manager's last AIMessage. Only the first matching call
        # is recoverable this way, which is why routing prefers Send.
        messages = state.get("messages") or []
        if not messages:
            raise RuntimeError(
                f"Worker '{worker_node_name}' was invoked with empty manager state."
            )
        last_ai = messages[-1]
        tool_calls = getattr(last_ai, "tool_calls", None) or []
        pending_call = next(
            (tc for tc in tool_calls if _tc_get(tc, "name") == delegate_tool_name),
            None,
        )
        if pending_call is None:
            raise RuntimeError(
                f"Worker '{worker_node_name}' was routed to but the manager's "
                f"last message has no '{delegate_tool_name}' tool call."
            )
        args = _tc_get(pending_call, "args") or {}
        call_id = _tc_get(pending_call, "id") or ""
        return args.get("task") or "", call_id

    def _tool_message_from(reply: str, call_id: str) -> Dict[str, Any]:
        return {"messages": [ToolMessage(content=reply, tool_call_id=call_id)]}

    def _worker_input(task: str) -> Dict[str, Any]:
        # Content isolation: the worker is fed ONLY the delegated task, never
        # the manager's history. We deliberately pass NO explicit config so
        # the worker run *inherits* the ambient run config of this node —
        # which carries (a) the astream_events callbacks and (b) this node's
        # ``checkpoint_ns`` (``<worker_node>:<task_id>``). Inheriting the
        # namespace is what makes the worker's token events stream natively
        # under the worker node (so consumers can attribute them to the
        # worker) instead of escaping to a detached ``agent:<uuid>`` run.
        #
        # State stays isolated *across* delegations without a fresh thread_id:
        # LangGraph gives each ``<worker_node>`` invocation a distinct
        # per-superstep ``checkpoint_ns``, so a worker called twice in a row
        # starts each run fresh rather than replaying its previous answer.
        #
        # The task is forwarded as a SystemMessage, not a HumanMessage: it is
        # the manager's internal instruction to the worker, not something the
        # end user typed. Because the worker inherits this node's
        # astream_events callbacks (above), its input message streams out to
        # consumers — and a HumanMessage there surfaces in the chat UI as a
        # spurious end-user turn. A SystemMessage still drives the worker while
        # being rendered/attributed as an instruction rather than a user turn.
        return {"messages": [SystemMessage(content=task)]}

    def _last_message_content(result: Any) -> str:
        messages = result.get("messages") if isinstance(result, dict) else None
        if not messages:
            return ""
        return getattr(messages[-1], "content", "") or ""

    def _error_reply(call_id: str, exc: Exception) -> Dict[str, Any]:
        # A worker crash must not abort the whole parent run and leave the
        # manager's ``delegate_to_<worker>`` tool-call unanswered: an orphan
        # tool-call breaks the OpenAI/Anthropic contract and 400s the manager's
        # next turn (in particular on a checkpoint resume). Answer the pending
        # delegation with an error ToolMessage instead, so the manager sees a
        # well-formed "worker failed" tool response and can decide how to react
        # — mirroring how a tool raising inside the react-agent ToolNode is
        # surfaced as an error ToolMessage rather than crashing the graph.
        logging.getLogger("pyagentspec.adapters.langgraph").exception(
            "Worker '%s' failed; answering delegation %s with an error ToolMessage",
            worker_node_name,
            call_id,
        )
        return {
            "messages": [
                ToolMessage(
                    content=f"Worker '{worker_node_name}' failed: {exc}",
                    tool_call_id=call_id,
                    status="error",
                )
            ]
        }

    def _run_sync(state: Dict[str, Any]) -> Dict[str, Any]:
        # ``_extract_pending`` runs outside the try: if it raises there is no
        # pending delegation to answer, so the error must surface unchanged.
        task, call_id = _extract_pending(state)
        try:
            result = worker_graph.invoke(_worker_input(task))
        except Exception as exc:  # noqa: BLE001 — degrade any worker failure
            return _error_reply(call_id, exc)
        return _tool_message_from(_last_message_content(result), call_id)

    async def _run_async(state: Dict[str, Any]) -> Dict[str, Any]:
        task, call_id = _extract_pending(state)
        try:
            result = await worker_graph.ainvoke(_worker_input(task))
        except Exception as exc:  # noqa: BLE001 — degrade any worker failure
            return _error_reply(call_id, exc)
        return _tool_message_from(_last_message_content(result), call_id)

    return RunnableLambda(
        func=_run_sync,
        afunc=_run_async,
        name=f"worker:{worker_node_name}",
    )


def _patch_with_manager_workers_execution_span(
    compiled_graph: CompiledStateGraph[Any, Any, Any],
    mw: AgentSpecManagerWorkers,
) -> None:
    """Wrap ``stream`` / ``astream`` so each ManagerWorkers run emits a
    ``ManagerWorkersExecutionSpan`` with Start/End events. Mirrors the
    patches applied to Agent and Flow compiled graphs elsewhere in this
    converter.
    """
    original_stream = compiled_graph.stream
    original_astream = compiled_graph.astream

    def _coerce_inputs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        inputs = kwargs.get("input", {})
        return inputs if isinstance(inputs, dict) else {}

    def patched_stream(*args: Any, **kwargs: Any) -> Generator[Any, Any, None]:
        span_name = f"ManagerWorkersExecution[{mw.name}]"
        inputs = _coerce_inputs(kwargs)
        with AgentSpecManagerWorkersExecutionSpan(name=span_name, managerworkers=mw) as span:
            span.add_event(
                AgentSpecManagerWorkersExecutionStart(managerworkers=mw, inputs=inputs)
            )
            last_chunk: Dict[str, Any] = {}
            for chunk in original_stream(*args, **kwargs):
                yield chunk
                if isinstance(chunk, tuple) and isinstance(chunk[1], dict):
                    last_chunk = chunk[1]
            span.add_event(
                AgentSpecManagerWorkersExecutionEnd(
                    managerworkers=mw,
                    outputs={"messages": last_chunk.get("messages", [])},
                )
            )

    async def patched_astream(*args: Any, **kwargs: Any) -> AsyncGenerator[Any, Any]:
        span_name = f"ManagerWorkersExecution[{mw.name}]"
        inputs = _coerce_inputs(kwargs)
        span = AgentSpecManagerWorkersExecutionSpan(name=span_name, managerworkers=mw)
        try:
            await span.start_async()
        except NotImplementedError:
            span.start()
        try:
            start_event = AgentSpecManagerWorkersExecutionStart(
                managerworkers=mw, inputs=inputs
            )
            try:
                await span.add_event_async(start_event)
            except NotImplementedError:
                span.add_event(start_event)
            last_chunk: Dict[str, Any] = {}
            async for chunk in original_astream(*args, **kwargs):
                yield chunk
                if isinstance(chunk, tuple) and isinstance(chunk[1], dict):
                    last_chunk = chunk[1]
            end_event = AgentSpecManagerWorkersExecutionEnd(
                managerworkers=mw,
                outputs={"messages": last_chunk.get("messages", [])},
            )
            try:
                await span.add_event_async(end_event)
            except NotImplementedError:
                span.add_event(end_event)
        finally:
            try:
                await span.end_async()
            except NotImplementedError:
                span.end()

    compiled_graph.stream = patched_stream  # type: ignore[assignment]
    compiled_graph.astream = patched_astream  # type: ignore[assignment]
