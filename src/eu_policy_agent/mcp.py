"""MCP (Model Context Protocol) integration utilities for EU Policy Agent.

Provides thin wrappers around DatabricksMCPClient to convert Databricks-managed
MCP tool definitions into the ToolInfo format expected by EuPolicyAgent.

Supported MCP servers:
- Vector Search:  {host}/api/2.0/mcp/vector-search/{catalog}/{schema}
- Genie Space:    {host}/api/2.0/mcp/genie/{genie_space_id}  (optional)

Usage:
    tools = asyncio.run(create_mcp_tools(w, [vs_url]))
    agent = EuPolicyAgent(..., extra_tools=tools)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from pydantic import BaseModel


class ToolInfo(BaseModel):
    """Descriptor for a single agent tool.

    Attributes:
        name: Unique tool name (used as key in the agent's tools dict).
        spec: Full OpenAI Responses-compatible tool specification dict.
        exec_fn: Callable that implements the tool; receives ``**kwargs``
            matching the spec's parameter schema and returns a string.
    """

    name: str
    spec: dict[str, Any]
    exec_fn: Callable[..., Any]

    model_config = {"arbitrary_types_allowed": True}


def create_managed_exec_fn(
    server_url: str,
    tool_name: str,
    w: WorkspaceClient,
) -> Callable[..., str]:
    """Return an execution function that delegates to a Databricks MCP tool.

    A new ``DatabricksMCPClient`` is created per call so that the function
    is safe to pickle and does not hold a stale connection.

    Args:
        server_url: Full URL of the MCP server endpoint.
        tool_name: Name of the tool on that server.
        w: Authenticated ``WorkspaceClient``.

    Returns:
        A ``**kwargs``-accepting callable that returns the tool output as a
        plain string.
    """

    def exec_fn(**kwargs: Any) -> str:
        client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        response = client.call_tool(tool_name, kwargs)
        return "".join(chunk.text for chunk in response.content)

    return exec_fn


async def create_mcp_tools(
    w: WorkspaceClient,
    url_list: list[str],
) -> list[ToolInfo]:
    """Discover and wrap all tools exposed by a list of MCP servers.

    For each URL, lists available tools via the MCP protocol and converts
    them into ``ToolInfo`` objects ready for use in ``EuPolicyAgent``.

    Args:
        w: Authenticated ``WorkspaceClient`` (provides workspace auth).
        url_list: MCP server URLs to enumerate. Empty or None entries are
            skipped so callers can pass ``cfg.genie_space_id`` conditionally.

    Returns:
        List of ``ToolInfo`` objects, one per discovered tool across all
        servers. Tools are ordered by server, then by the server's listing
        order.
    """
    tools: list[ToolInfo] = []
    for server_url in url_list:
        if not server_url:
            continue
        mcp_client = DatabricksMCPClient(server_url=server_url, workspace_client=w)
        for mcp_tool in mcp_client.list_tools():
            input_schema: dict[str, Any] = (
                mcp_tool.inputSchema.copy() if mcp_tool.inputSchema else {}
            )
            tool_spec: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": mcp_tool.name,
                    "parameters": input_schema,
                    "description": mcp_tool.description or f"Tool: {mcp_tool.name}",
                },
            }
            tools.append(
                ToolInfo(
                    name=mcp_tool.name,
                    spec=tool_spec,
                    exec_fn=create_managed_exec_fn(server_url, mcp_tool.name, w),
                )
            )
    return tools
