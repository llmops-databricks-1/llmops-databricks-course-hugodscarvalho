"""EU Policy Agent — production-grade LLM agent over EU legislation.

Architecture
------------
``EuPolicyAgent`` extends MLflow's ``ResponsesAgent`` base class and follows
the OpenAI Responses API contract.  It integrates:

* **MCP tools** — Databricks Vector Search index over EU policy chunks, with
  an optional Genie Space for analytics queries.
* **MLflow tracing** — every logical step is wrapped in a typed span so traces
  are navigable in the Databricks MLflow UI.
* **Lakebase memory** — optional per-session conversation persistence backed
  by managed PostgreSQL, enabling multi-turn interactions.

Tracing hierarchy
-----------------
    predict_stream()          → AGENT span (root)
      load_memory()           → RETRIEVER span
      call_and_run_tools()    → CHAIN span
        call_llm()            → LLM span   (per LLM call, via start_span)
        execute_tool()        → TOOL span   (per tool invocation)
      save_memory()           → CHAIN span

Deployment
----------
See ``eu_policy_agent.py`` at the repo root for the MLflow model-serving
entry point.  Use ``log_register_agent()`` from notebook 4.3 to register
a versioned model to Unity Catalog.
"""

from __future__ import annotations

import asyncio
import json
import os
import warnings
from collections.abc import Generator
from datetime import datetime
from typing import Any
from uuid import uuid4

import backoff
import mlflow
import nest_asyncio
import openai
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow import MlflowClient
from mlflow.entities import SpanType
from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksVectorSearchIndex,
)
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)

from eu_policy_agent.config import ProjectConfig
from eu_policy_agent.mcp import ToolInfo, create_mcp_tools
from eu_policy_agent.memory import LakebaseMemory


class EuPolicyAgent(ResponsesAgent):
    """AI agent that answers questions over EU legislation documents.

    The agent uses Vector Search (via MCP) to retrieve relevant chunks from
    the EU policy corpus, passes them as tool results to the LLM, and
    synthesises a grounded answer that cites specific regulations.

    Args:
        llm_endpoint: Databricks model serving endpoint name.
        system_prompt: Instruction prompt prepended to every conversation.
        catalog: Unity Catalog catalog name (used to build MCP URLs).
        schema: Unity Catalog schema name (used to build MCP URLs).
        genie_space_id: Optional Genie Space ID for analytics queries.
            Pass ``None`` or empty string to disable Genie tooling.
        lakebase_project_id: Optional Lakebase project ID for session memory.
            Pass ``None`` to run stateless (no persistent memory).
    """

    def __init__(
        self,
        llm_endpoint: str,
        system_prompt: str,
        catalog: str,
        schema: str,
        genie_space_id: str | None = None,
        lakebase_project_id: str | None = None,
    ) -> None:
        nest_asyncio.apply()

        self.system_prompt = system_prompt
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )

        # Lakebase session memory (optional)
        self.memory: LakebaseMemory | None = (
            LakebaseMemory(project_id=lakebase_project_id)
            if lakebase_project_id
            else None
        )

        # Build MCP tool URLs — Vector Search is always included; Genie is
        # optional and skipped when the ID is empty/None.
        host = self.workspace_client.config.host
        mcp_urls = [f"{host}/api/2.0/mcp/vector-search/{catalog}/{schema}"]
        if genie_space_id:
            mcp_urls.append(f"{host}/api/2.0/mcp/genie/{genie_space_id}")

        tools: list[ToolInfo] = asyncio.run(
            create_mcp_tools(w=self.workspace_client, url_list=mcp_urls)
        )
        self._tools_dict: dict[str, ToolInfo] = {t.name: t for t in tools}
        logger.info(
            f"EuPolicyAgent initialised with {len(self._tools_dict)} tool(s): "
            f"{list(self._tools_dict)}"
        )

    # Tool interface

    def get_tool_specs(self) -> list[dict[str, Any]]:
        """Return tool specifications in the OpenAI Responses format."""
        return [t.spec for t in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Execute a tool by name and return its raw output.

        Args:
            tool_name: Name of the registered tool.
            args: Keyword arguments to pass to the tool.

        Returns:
            Tool output (stringified before injection into message history).
        """
        if tool_name not in self._tools_dict:
            raise ValueError(f"Unknown tool: {tool_name!r}")
        return self._tools_dict[tool_name].exec_fn(**args)

    # LLM call

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def call_llm(
        self,
        messages: list[dict[str, Any]],
    ) -> Generator[dict[str, Any], None, None]:
        """Stream a chat-completion request to the configured LLM endpoint.

        Wraps the streaming response in an LLM span and records model name,
        token usage, and the upstream request ID for debugging.

        Args:
            messages: Full message history (system + conversation + tool
                results) in OpenAI chat format.

        Yields:
            Raw chunk dicts from the streaming response.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="PydanticSerializationUnexpectedValue"
            )
            stream = self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=to_chat_completions_input(messages),
                tools=self.get_tool_specs(),
                stream=True,
            )
            with mlflow.start_span(name="call_llm", span_type=SpanType.LLM) as span:
                last_chunk: dict[str, Any] = {}
                for chunk in stream:
                    chunk_dict = chunk.to_dict()
                    last_chunk = chunk_dict
                    yield chunk_dict

                llm_request_id = stream.response.headers.get("x-request-id")
                outputs: dict[str, Any] = {
                    "model": last_chunk.get("model"),
                    "usage": last_chunk.get("usage"),
                }
                if llm_request_id:
                    outputs["llm_request_id"] = llm_request_id
                span.set_outputs(outputs)

    # Tool-call handling

    def handle_tool_call(
        self,
        tool_call: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> ResponsesAgentStreamEvent:
        """Execute a single tool call and inject the result into messages.

        Args:
            tool_call: Tool call descriptor dict from the LLM response.
            messages: Running message list (mutated in place).

        Returns:
            A ``ResponsesAgentStreamEvent`` wrapping the tool output.
        """
        args = json.loads(tool_call["arguments"])
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))
        output_item = self.create_function_call_output_item(tool_call["call_id"], result)
        messages.append(output_item)
        return ResponsesAgentStreamEvent(
            type="response.output_item.done", item=output_item
        )

    # Memory

    @mlflow.trace(span_type=SpanType.RETRIEVER, name="memory_load")
    def load_memory(self, session_id: str) -> list[dict[str, Any]]:
        """Load previous messages for ``session_id`` from Lakebase.

        Returns an empty list when memory is not configured.
        """
        if self.memory:
            return self.memory.load_messages(session_id)
        return []

    @mlflow.trace(span_type=SpanType.CHAIN, name="memory_save")
    def save_memory(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        """Persist new messages for ``session_id`` to Lakebase."""
        if self.memory:
            self.memory.save_messages(session_id, messages)

    # Agent loop

    def _extract_output_items(
        self, events: list[ResponsesAgentStreamEvent]
    ) -> list[dict[str, Any]]:
        """Extract serialised message items from stream events."""
        items = []
        for event in events:
            if event.type != "response.output_item.done":
                continue
            item = event.item if isinstance(event.item, dict) else event.item.model_dump()
            if item.get("type") == "message":
                items.append(item)
        return items

    def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 10,
    ) -> list[ResponsesAgentStreamEvent]:
        """Drive the LLM ↔ tool loop until the model stops or max_iter.

        Iterates by inspecting the last message in the running history:
        - If it's a final assistant message → stop.
        - If it's a function_call → execute the tool.
        - Otherwise → call the LLM and extend the history.

        Args:
            messages: Mutable message list (system + history + user input).
            max_iter: Safety ceiling to prevent infinite loops.

        Returns:
            Ordered list of stream events produced during the loop.
        """
        events: list[ResponsesAgentStreamEvent] = []
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role") == "assistant":
                break
            elif last_msg.get("type") == "function_call":
                events.append(self.handle_tool_call(last_msg, messages))
            else:
                events.extend(
                    output_to_responses_items_stream(
                        chunks=self.call_llm(messages),
                        aggregator=messages,
                    )
                )
        else:
            # max_iter reached — emit a graceful stop message
            events.append(
                ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(
                        "Maximum reasoning steps reached. Please refine your question.",
                        str(uuid4()),
                    ),
                )
            )
        return events

    @mlflow.trace(span_type=SpanType.CHAIN)
    def call_and_run_tools(
        self,
        request_input: list[dict[str, Any]],
        previous_messages: list[dict[str, Any]] | None = None,
        request_id: str | None = None,
        session_id: str | None = None,
    ) -> list[ResponsesAgentStreamEvent]:
        """Build context, run the tool loop, and optionally save memory.

        Attaches deployment metadata (git SHA, endpoint name, model version)
        and session/request IDs to the active MLflow trace for downstream
        filtering and debugging.

        Args:
            request_input: Current-turn messages from the user.
            previous_messages: Prior conversation messages loaded from memory.
            request_id: Unique ID for this individual request.
            session_id: Conversation session identifier for memory and tracing.

        Returns:
            Stream events produced by the tool loop.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        if previous_messages:
            messages.extend(previous_messages)
        messages.extend(request_input)

        # Attach observability metadata to the current trace
        mlflow.update_current_trace(
            tags={
                "git_sha": os.getenv("GIT_SHA", "local"),
                "model_serving_endpoint_name": os.getenv(
                    "MODEL_SERVING_ENDPOINT_NAME", "local"
                ),
                "model_version": os.getenv("MODEL_VERSION", "local"),
            },
            metadata=({"mlflow.trace.session": session_id} if session_id else {}),
            client_request_id=request_id,
        )

        events = self._run_tool_loop(messages)

        if session_id and self.memory:
            self.save_memory(
                session_id,
                request_input + self._extract_output_items(events),
            )
        return events

    # ResponsesAgent interface

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Synchronous predict — collect full response before returning."""
        events = list(self.predict_stream(request))
        return ResponsesAgentResponse(
            output=self._extract_output_items(events),
            custom_outputs=request.custom_inputs,
        )

    @mlflow.trace(span_type=SpanType.AGENT)
    def predict_stream(
        self, request: ResponsesAgentRequest
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming predict — root AGENT span wraps the full request.

        Reads ``session_id`` and ``request_id`` from ``custom_inputs`` if
        provided, loads prior session memory, runs the tool loop, and yields
        stream events.

        Args:
            request: Incoming ``ResponsesAgentRequest``.

        Yields:
            ``ResponsesAgentStreamEvent`` objects as they are produced.
        """
        custom = request.custom_inputs or {}
        session_id: str | None = custom.get("session_id")
        request_id: str | None = custom.get("request_id")

        previous_messages = (
            self.load_memory(session_id) if session_id and self.memory else []
        )

        request_input = [item.model_dump() for item in request.input]
        events = self.call_and_run_tools(
            request_input=request_input,
            previous_messages=previous_messages,
            request_id=request_id,
            session_id=session_id,
        )
        yield from events


# Model registration utility


def log_register_agent(
    cfg: ProjectConfig,
    git_sha: str,
    run_id: str,
    agent_code_path: str,
    model_name: str,
    evaluation_metrics: dict[str, float] | None = None,
) -> mlflow.entities.model_registry.RegisteredModel:
    """Log and register the EU Policy Agent model to Unity Catalog.

    Declares all Databricks resources that the agent depends on so that
    Model Serving can provision the right service identity and permissions.

    Args:
        cfg: Resolved ``ProjectConfig`` for the target environment.
        git_sha: Git commit SHA — stored as a tag for traceability.
        run_id: Deployment run ID — stored as a tag.
        agent_code_path: Absolute or relative path to ``agent_serving.py``.
        model_name: Fully-qualified UC model path
            (``{catalog}.{schema}.eu_policy_agent``).
        evaluation_metrics: Optional dict of evaluation metrics to log with
            the MLflow run (e.g. ``{"word_count_pass_rate": 0.92}``).

    Returns:
        The ``RegisteredModel`` object from MLflow / Unity Catalog.
    """
    resources = [
        DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
        DatabricksServingEndpoint(endpoint_name=cfg.embedding_endpoint),
        DatabricksVectorSearchIndex(
            index_name=f"{cfg.catalog}.{cfg.schema}.eu_policy_index"
        ),
        DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.eu_policy_chunks"),
        DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.raw_documents"),
        DatabricksSQLWarehouse(warehouse_id=cfg.warehouse_id),
    ]
    if cfg.genie_space_id:
        resources.append(DatabricksGenieSpace(genie_space_id=cfg.genie_space_id))

    model_config = {
        "catalog": cfg.catalog,
        "schema": cfg.schema,
        "genie_space_id": cfg.genie_space_id or "",
        "system_prompt": cfg.system_prompt,
        "llm_endpoint": cfg.llm_endpoint,
        "lakebase_project_id": cfg.lakebase_project_id or "",
    }

    # Representative test request for the model signature
    test_request = {
        "input": [
            {
                "role": "user",
                "content": (
                    "What are the main obligations for high-risk AI system providers "
                    "under the EU AI Act?"
                ),
            }
        ]
    }

    experiment_path = cfg.experiment_path or f"/Shared/{model_name}"
    mlflow.set_experiment(experiment_path)
    ts = datetime.now().strftime("%Y-%m-%d")

    with mlflow.start_run(
        run_name=f"eu-policy-agent-{ts}",
        tags={"git_sha": git_sha, "run_id": run_id},
    ):
        model_info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=agent_code_path,
            resources=resources,
            input_example=test_request,
            model_config=model_config,
        )
        if evaluation_metrics:
            mlflow.log_metrics(evaluation_metrics)

    logger.info(f"Registering model: {model_name}")
    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=model_name,
        env_pack="databricks_model_serving",
        tags={"git_sha": git_sha, "run_id": run_id},
    )
    logger.info(f"Registered version: {registered_model.version}")

    client = MlflowClient()
    logger.info("Setting alias 'champion'")
    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=registered_model.version,
    )
    return registered_model
