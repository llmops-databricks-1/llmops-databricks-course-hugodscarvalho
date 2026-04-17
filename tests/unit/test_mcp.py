"""Unit tests for eu_policy_agent.mcp.

Tests cover:
- ToolInfo model construction and arbitrary callable acceptance
- create_managed_exec_fn: DatabricksMCPClient interaction
- create_mcp_tools: async tool discovery, empty URL filtering

Note: async tests are run via asyncio.run() to avoid a pytest-asyncio
dependency that is not needed in the CI extras.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from eu_policy_agent.mcp import ToolInfo, create_managed_exec_fn, create_mcp_tools

# ToolInfo


class TestToolInfo:
    def test_construction_with_callable(self) -> None:
        def dummy(**kwargs: object) -> str:
            return "ok"

        tool = ToolInfo(
            name="my_tool",
            spec={"type": "function", "function": {"name": "my_tool"}},
            exec_fn=dummy,
        )
        assert tool.name == "my_tool"
        assert tool.exec_fn() == "ok"

    def test_exec_fn_passes_kwargs(self) -> None:
        received: list[dict] = []

        def capture(**kwargs: object) -> str:
            received.append(kwargs)
            return "captured"

        tool = ToolInfo(name="t", spec={}, exec_fn=capture)
        tool.exec_fn(query="test", num=3)
        assert received == [{"query": "test", "num": 3}]

    def test_spec_is_stored_as_is(self) -> None:
        spec = {"type": "function", "function": {"name": "t", "description": "desc"}}
        tool = ToolInfo(name="t", spec=spec, exec_fn=lambda: None)
        assert tool.spec == spec


# create_managed_exec_fn


class TestCreateManagedExecFn:
    def test_calls_mcp_client_and_joins_text(self) -> None:
        mock_workspace = MagicMock()

        chunk1 = MagicMock()
        chunk1.text = "Hello "
        chunk2 = MagicMock()
        chunk2.text = "World"
        fake_response = MagicMock()
        fake_response.content = [chunk1, chunk2]

        with patch("eu_policy_agent.mcp.DatabricksMCPClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.call_tool.return_value = fake_response
            mock_client_cls.return_value = mock_client

            exec_fn = create_managed_exec_fn(
                server_url="https://example.databricks.com/api/2.0/mcp/vs/cat/sch",
                tool_name="query",
                w=mock_workspace,
            )
            result = exec_fn(query="GDPR rights", num_results=3)

        assert result == "Hello World"
        mock_client.call_tool.assert_called_once_with(
            "query", {"query": "GDPR rights", "num_results": 3}
        )

    def test_creates_new_client_per_call(self) -> None:
        mock_workspace = MagicMock()
        fake_response = MagicMock()
        fake_response.content = []

        with patch("eu_policy_agent.mcp.DatabricksMCPClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.call_tool.return_value = fake_response
            mock_client_cls.return_value = mock_client

            exec_fn = create_managed_exec_fn("url", "tool", mock_workspace)
            exec_fn()
            exec_fn()

        # A new client should be created for each call
        assert mock_client_cls.call_count == 2


# create_mcp_tools


def _make_mcp_tool(name: str, description: str, schema: dict | None = None) -> MagicMock:
    t = MagicMock()
    t.name = name
    t.description = description
    t.inputSchema = schema or {"type": "object", "properties": {}}
    return t


class TestCreateMcpTools:
    def test_returns_tool_info_per_discovered_tool(self) -> None:
        mock_workspace = MagicMock()
        fake_tool = _make_mcp_tool("query", "Search vector index")

        with patch("eu_policy_agent.mcp.DatabricksMCPClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.list_tools.return_value = [fake_tool]
            mock_cls.return_value = mock_client

            tools = asyncio.run(
                create_mcp_tools(
                    w=mock_workspace,
                    url_list=["https://host/api/2.0/mcp/vs/cat/sch"],
                )
            )

        assert len(tools) == 1
        assert tools[0].name == "query"
        assert tools[0].spec["function"]["description"] == "Search vector index"

    def test_empty_urls_are_skipped(self) -> None:
        mock_workspace = MagicMock()

        with patch("eu_policy_agent.mcp.DatabricksMCPClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.list_tools.return_value = []
            mock_cls.return_value = mock_client

            tools = asyncio.run(
                create_mcp_tools(
                    w=mock_workspace,
                    url_list=["", None, ""],  # type: ignore[list-item]
                )
            )

        assert tools == []
        mock_cls.assert_not_called()

    def test_aggregates_tools_from_multiple_servers(self) -> None:
        mock_workspace = MagicMock()
        tool_a = _make_mcp_tool("tool_a", "Tool A")
        tool_b = _make_mcp_tool("tool_b", "Tool B")

        call_count = [0]

        def client_factory(**kwargs: object) -> MagicMock:
            client = MagicMock()
            server_url = str(kwargs.get("server_url", ""))
            if "vs" in server_url:
                client.list_tools.return_value = [tool_a]
            else:
                client.list_tools.return_value = [tool_b]
            call_count[0] += 1
            return client

        with patch("eu_policy_agent.mcp.DatabricksMCPClient", side_effect=client_factory):
            tools = asyncio.run(
                create_mcp_tools(
                    w=mock_workspace,
                    url_list=[
                        "https://host/api/2.0/mcp/vs/cat/sch",
                        "https://host/api/2.0/mcp/genie/abc123",
                    ],
                )
            )

        assert len(tools) == 2
        assert {t.name for t in tools} == {"tool_a", "tool_b"}

    def test_falls_back_to_generic_description_when_none(self) -> None:
        mock_workspace = MagicMock()
        no_desc_tool = _make_mcp_tool("nameless", "")
        no_desc_tool.description = None

        with patch("eu_policy_agent.mcp.DatabricksMCPClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.list_tools.return_value = [no_desc_tool]
            mock_cls.return_value = mock_client

            tools = asyncio.run(
                create_mcp_tools(
                    w=mock_workspace,
                    url_list=["https://host/api/2.0/mcp/vs/cat/sch"],
                )
            )

        assert "nameless" in tools[0].spec["function"]["description"]
