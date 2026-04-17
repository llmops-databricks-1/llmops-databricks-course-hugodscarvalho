# Databricks notebook source
"""
Week 3 — MCP Integration Deep Dive
====================================
Topics covered:
  - MCP protocol fundamentals: servers, clients, tools, resources
  - Databricks-managed MCP servers (Vector Search, Genie Space)
  - Connecting to MCP servers with DatabricksMCPClient
  - Listing and calling tools directly via the MCP protocol
  - ToolInfo / create_mcp_tools from eu_policy_agent.mcp
  - Combining multiple MCP servers into one tool registry
  - SimpleAgent: minimal agent loop driven by MCP tools
  - MCP vs. custom Python functions — trade-offs

This notebook complements 3.1_agent_tool_calling.py by going deeper
into the MCP protocol itself and showing how DatabricksMCPClient works
before the eu_policy_agent package abstracts it away.
"""

# COMMAND ----------

import asyncio
import json
import os

import nest_asyncio
from databricks.sdk import WorkspaceClient
from databricks_mcp import DatabricksMCPClient
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from eu_policy_agent.config import get_env, load_config
from eu_policy_agent.mcp import ToolInfo, create_mcp_tools

# Enable nested event loops (required for Databricks notebooks)
nest_asyncio.apply()

# COMMAND ----------
# Environment setup

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ["PROFILE"]

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()
host = w.config.host

logger.info(f"Environment   : {env}")
logger.info(f"Catalog/schema: {cfg.catalog}.{cfg.schema}")
logger.info(f"Workspace host: {host}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — What is Model Context Protocol (MCP)?
# MAGIC
# MAGIC **MCP** is an open standard for connecting AI models to external data sources
# MAGIC and tools. Instead of each team writing bespoke tool integrations, MCP provides
# MAGIC a single protocol that any LLM framework can consume.
# MAGIC
# MAGIC ### Key components
# MAGIC
# MAGIC | Component | Role |
# MAGIC |---|---|
# MAGIC | **MCP Server** | Exposes tools and resources over HTTP/SSE |
# MAGIC | **MCP Client** | Connects to a server, lists tools, calls them |
# MAGIC | **Tool** | A callable function with a schema (name, description, parameters) |
# MAGIC | **Resource** | Read-only data the model can access (files, DB rows, …) |
# MAGIC
# MAGIC ### Why use Databricks Managed MCP?
# MAGIC
# MAGIC - **No infrastructure** — Databricks runs the servers; you just provide a URL.
# MAGIC - **Auto-discovery** — tools appear automatically as you create Vector Search
# MAGIC   indexes, Genie Spaces, or UC Functions.
# MAGIC - **Built-in auth** — workspace credentials flow through transparently.
# MAGIC - **Enterprise SLAs** — HA, audit logs, rate limiting included.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — MCP vs. Custom Python Functions
# MAGIC
# MAGIC | Aspect | Databricks MCP | Custom Python function |
# MAGIC |---|---|---|
# MAGIC | Setup effort | URL only | Write spec + implementation |
# MAGIC | Maintenance | Databricks-managed | You maintain |
# MAGIC | Reusability | Any agent across workspace | Per-agent |
# MAGIC | Custom logic | Limited to what MCP exposes | Unlimited |
# MAGIC | Auth | Automatic | Manual |
# MAGIC | Best for | Standard ops (search, SQL, UC) | Domain-specific bespoke logic |
# MAGIC
# MAGIC **Use MCP for:** Vector Search retrieval, Genie natural-language analytics,
# MAGIC Unity Catalog function execution.
# MAGIC
# MAGIC **Use custom functions for:** proprietary APIs, complex business logic,
# MAGIC multi-step transformations that MCP doesn't expose.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — Databricks MCP Servers
# MAGIC
# MAGIC | Server | URL pattern | Auto-exposes |
# MAGIC |---|---|---|
# MAGIC | Vector Search | `{host}/api/2.0/mcp/vector-search/{catalog}/{schema}` | One tool per VS index in the schema |
# MAGIC | Genie Space | `{host}/api/2.0/mcp/genie/{genie_space_id}` | `ask_genie` tool |
# MAGIC | UC Functions | `{host}/api/2.0/mcp/unity-catalog/functions` | One tool per UC function |
# MAGIC | DBSQL | `{host}/api/2.0/mcp/sql` | SQL execution tool |

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — Vector Search MCP

# COMMAND ----------

# Build the Vector Search MCP URL
vs_mcp_url = f"{host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}"
logger.info(f"Vector Search MCP URL: {vs_mcp_url}")

# COMMAND ----------
# Connect and list tools exposed by the Vector Search MCP server

vs_client = DatabricksMCPClient(server_url=vs_mcp_url, workspace_client=w)
vs_tools = vs_client.list_tools()

logger.info(f"Vector Search MCP exposes {len(vs_tools)} tool(s):")
logger.info("=" * 80)
for tool in vs_tools:
    logger.info(f"  Tool  : {tool.name}")
    logger.info(f"  Desc  : {tool.description}")
    if tool.inputSchema:
        params = list(tool.inputSchema.get("properties", {}).keys())
        logger.info(f"  Params: {params}")
    logger.info("")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Tool naming convention
# MAGIC
# MAGIC The Vector Search MCP server names each tool after its index:
# MAGIC ```
# MAGIC {catalog}__{schema}__{index_name}
# MAGIC ```
# MAGIC — double underscores separate catalog, schema, and index.
# MAGIC
# MAGIC For our project:
# MAGIC ```
# MAGIC dev__eu_policy__eu_policy_index
# MAGIC ```
# MAGIC The single parameter is `query` — the search text. The embedding and
# MAGIC similarity search happen server-side.

# COMMAND ----------
# Call the Vector Search tool directly (raw MCP protocol)

tool_name = f"{cfg.catalog}__{cfg.schema}__eu_policy_index"

raw_result = vs_client.call_tool(
    tool_name,
    {"query": "What are the obligations of controllers under GDPR?"},
)

logger.info("Raw MCP tool result (first 500 chars):")
for content in raw_result.content:
    logger.info(content.text[:500])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5 — Genie Space MCP (optional)
# MAGIC
# MAGIC Genie turns natural-language questions into SQL, executes them against
# MAGIC a configured set of Delta tables, and returns structured results.
# MAGIC It is useful for **analytics queries** over your EU policy dataset —
# MAGIC e.g., "How many chunks are from the AI Act?", "Which regulations have
# MAGIC the most articles?"
# MAGIC
# MAGIC > **Setup:** create a Genie Space in the Databricks UI, then paste the
# MAGIC > Space ID into `genie_space_id` in `project_config.yml`.

# COMMAND ----------

if cfg.genie_space_id:
    genie_mcp_url = f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}"
    logger.info(f"Genie Space MCP URL: {genie_mcp_url}")

    genie_client = DatabricksMCPClient(server_url=genie_mcp_url, workspace_client=w)
    genie_tools = genie_client.list_tools()

    logger.info(f"Genie Space MCP exposes {len(genie_tools)} tool(s):")
    for tool in genie_tools:
        logger.info(f"  Tool: {tool.name}")
        logger.info(f"  Desc: {tool.description}")
else:
    logger.warning(
        "Genie Space not configured — set genie_space_id in project_config.yml "
        "and re-run to see Genie MCP tools."
    )
    logger.info("See notebook 3.4_genie_space.py for setup instructions.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6 — Diagnosing MCP Connectivity

# COMMAND ----------


def test_mcp_connection(mcp_url: str, label: str) -> bool:
    """Test whether an MCP server is reachable and lists tools.

    Args:
        mcp_url: Full MCP server URL.
        label: Human-readable label for logging.

    Returns:
        True if the connection succeeded.
    """
    try:
        client = DatabricksMCPClient(server_url=mcp_url, workspace_client=w)
        tools = client.list_tools()
        logger.info(f"✓ {label}: connected — {len(tools)} tool(s) available")
        return True
    except Exception as exc:
        logger.error(f"✗ {label}: connection failed — {exc}")
        return False


test_mcp_connection(vs_mcp_url, "Vector Search MCP")

if cfg.genie_space_id:
    genie_url = f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}"
    test_mcp_connection(genie_url, "Genie Space MCP")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7 — eu_policy_agent.mcp Package Utilities
# MAGIC
# MAGIC The `eu_policy_agent.mcp` module wraps `DatabricksMCPClient` into the
# MAGIC `ToolInfo` format that `EuPolicyAgent` expects.
# MAGIC
# MAGIC | Symbol | Purpose |
# MAGIC |---|---|
# MAGIC | `ToolInfo` | Pydantic model: `name`, `spec` (OpenAI format), `exec_fn` |
# MAGIC | `create_managed_exec_fn()` | Returns a `**kwargs` callable that calls an MCP tool |
# MAGIC | `create_mcp_tools()` | Discovers all tools from a list of MCP URLs → `list[ToolInfo]` |

# COMMAND ----------
# Build full tool registry from all configured MCP servers

mcp_urls = [vs_mcp_url]
if cfg.genie_space_id:
    mcp_urls.append(f"{host}/api/2.0/mcp/genie/{cfg.genie_space_id}")

logger.info(f"Loading tools from {len(mcp_urls)} MCP server(s)…")

mcp_tools: list[ToolInfo] = asyncio.run(create_mcp_tools(w=w, url_list=mcp_urls))

logger.info(f"✓ Loaded {len(mcp_tools)} tool(s):")
for t in mcp_tools:
    logger.info(f"  • {t.name}")

# COMMAND ----------
# Inspect the OpenAI Responses API spec for each tool (what the LLM sees)

logger.info("Tool specifications (OpenAI Responses API format):")
logger.info("=" * 80)
for tool in mcp_tools:
    logger.info(json.dumps(tool.spec, indent=2))
    logger.info("")

# COMMAND ----------
# Call a tool via ToolInfo.exec_fn (the abstraction EuPolicyAgent uses)

if mcp_tools:
    vs_tool = mcp_tools[0]
    logger.info(f"Calling tool via ToolInfo.exec_fn: {vs_tool.name}")
    result = vs_tool.exec_fn(
        query="What are the key transparency requirements under the EU AI Act?"
    )
    logger.info(f"Result (first 500 chars):\n{str(result)[:500]}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8 — SimpleAgent: Minimal Agent Loop with MCP Tools
# MAGIC
# MAGIC A stripped-down agent that demonstrates the core loop without the
# MAGIC full EuPolicyAgent complexity. Useful for understanding the mechanics
# MAGIC before using the production class.

# COMMAND ----------


class SimpleAgent:
    """Minimal LLM agent that can call MCP tools in a loop.

    Implements the canonical tool-calling loop:
    1. Send messages + tool specs to the LLM.
    2. If the LLM returns tool calls → execute them, append results.
    3. Repeat until the LLM returns a final text response.

    Args:
        llm_endpoint: Databricks model serving endpoint name.
        system_prompt: System instruction prepended to every conversation.
        tools: List of ToolInfo objects (from create_mcp_tools).
    """

    def __init__(
        self,
        llm_endpoint: str,
        system_prompt: str,
        tools: list[ToolInfo],
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.system_prompt = system_prompt
        self._tools_dict = {t.name: t for t in tools}

        _w = WorkspaceClient()
        self._client = OpenAI(
            api_key=_w.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{_w.config.host}/serving-endpoints",
        )

    def get_tool_specs(self) -> list[dict]:
        """Return OpenAI-format tool specifications."""
        return [t.spec for t in self._tools_dict.values()]

    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name with given arguments."""
        if tool_name not in self._tools_dict:
            return f"Error: unknown tool '{tool_name}'"
        return str(self._tools_dict[tool_name].exec_fn(**args))

    def chat(self, user_message: str, max_iterations: int = 10) -> str:
        """Process a user message, allowing tool calls until a final answer.

        Args:
            user_message: The user's question.
            max_iterations: Safety ceiling to prevent infinite loops.

        Returns:
            The final assistant text response.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for _ in range(max_iterations):
            response = self._client.chat.completions.create(
                model=self.llm_endpoint,
                messages=messages,
                tools=self.get_tool_specs() or None,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                # No more tool calls — return the final answer
                return msg.content

            # Append the assistant turn (with tool calls)
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            # Execute each tool call and append results
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                logger.info(f"→ Tool call: {tc.function.name}({args})")
                result = self.execute_tool(tc.function.name, args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        return "Maximum iterations reached."


# COMMAND ----------
# Instantiate SimpleAgent with the discovered MCP tools

simple_agent = SimpleAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=(
        "You are an expert on EU digital legislation. "
        "Use the available tools to search for relevant regulatory text, "
        "then provide a clear, cited answer. "
        "Always cite the regulation and article when making claims."
    ),
    tools=mcp_tools,
)

logger.info("✓ SimpleAgent created")
logger.info(f"  Tools: {list(simple_agent._tools_dict.keys())}")

# COMMAND ----------

response = simple_agent.chat(
    "What are the main transparency obligations for providers of high-risk AI systems?"
)
logger.info("=" * 80)
logger.info(f"Answer:\n{response}")

# COMMAND ----------

response2 = simple_agent.chat(
    "What does the Digital Services Act require from intermediary service providers?"
)
logger.info("=" * 80)
logger.info(f"Answer:\n{response2}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9 — Key Takeaways
# MAGIC
# MAGIC 1. **MCP simplifies tool integration** — a URL + auth is all you need to
# MAGIC    connect an agent to Vector Search or Genie.
# MAGIC 2. **`create_mcp_tools()`** handles discovery, schema conversion, and
# MAGIC    execution function creation so `EuPolicyAgent` stays clean.
# MAGIC 3. **Tool specs matter** — the LLM decides *when* to call a tool based on
# MAGIC    the description alone. Good descriptions → correct tool selection.
# MAGIC 4. **Multiple MCP servers** can be combined into one tool registry by
# MAGIC    passing multiple URLs to `create_mcp_tools()`.
# MAGIC 5. **SimpleAgent** shows the loop; `EuPolicyAgent` adds tracing, memory,
# MAGIC    streaming, and production-grade error handling on top.
# MAGIC
# MAGIC → Next: `3.4_genie_space.py` for setting up the Genie Space that powers
# MAGIC the optional analytics tool in `EuPolicyAgent`.
