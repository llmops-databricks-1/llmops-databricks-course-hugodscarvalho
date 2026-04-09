# Databricks notebook source
"""
Week 3 — Agent Definition and Tool Calling
===========================================
Topics covered:
  - What agent tools are and why they matter
  - OpenAI Responses API tool specification format
  - ToolInfo and create_mcp_tools from eu_policy_agent.mcp
  - Vector Search as an MCP tool over EU legislation chunks
  - EuPolicyAgent: initialisation, tool listing, single-turn interaction
  - Best practices for tool design
"""

# COMMAND ----------

import asyncio
import json
import os

import mlflow
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from loguru import logger
from mlflow.types.responses import ResponsesAgentRequest
from pyspark.sql import SparkSession

from eu_policy_agent.config import get_env, load_config
from eu_policy_agent.mcp import ToolInfo, create_mcp_tools

# COMMAND ----------
# Environment / MLflow setup

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)
w = WorkspaceClient()

logger.info(f"Environment: {env}")
logger.info(f"Catalog / schema: {cfg.catalog}.{cfg.schema}")
logger.info(f"LLM endpoint: {cfg.llm_endpoint}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — Why Tools?
# MAGIC
# MAGIC LLMs alone cannot search databases, call APIs, or access current data.
# MAGIC **Tools bridge this gap** by giving the model the ability to take actions and
# MAGIC retrieve grounded information.
# MAGIC
# MAGIC For the EU Policy Agent the primary tool is **Vector Search** over the
# MAGIC `eu_policy_chunks` table — this is what turns a generic chat model into
# MAGIC a domain expert on EU digital legislation.
# MAGIC
# MAGIC Tool-calling flow:
# MAGIC ```
# MAGIC User: "What are GDPR's controller obligations?"
# MAGIC   ↓
# MAGIC Agent: decides to call vector_search(query="GDPR controller obligations")
# MAGIC   ↓
# MAGIC Tool: returns top-5 matching chunks from eu_policy_chunks
# MAGIC   ↓
# MAGIC Agent: synthesises answer citing the retrieved text
# MAGIC   ↓
# MAGIC Response: "Under GDPR Article 24 controllers must..."
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — Tool Specification Format (OpenAI Responses API)
# MAGIC
# MAGIC ```json
# MAGIC {
# MAGIC   "type": "function",
# MAGIC   "function": {
# MAGIC     "name": "search_eu_legislation",
# MAGIC     "description": "Search EU legislation chunks by semantic similarity",
# MAGIC     "parameters": {
# MAGIC       "type": "object",
# MAGIC       "properties": {
# MAGIC         "query": { "type": "string", "description": "..." }
# MAGIC       },
# MAGIC       "required": ["query"]
# MAGIC     }
# MAGIC   }
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC The description is the most important field — the LLM uses it to decide
# MAGIC **when** to call each tool.  Be precise and domain-specific.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — MCP Tools: Databricks Managed Vector Search

# COMMAND ----------

# Build the MCP server URL for Vector Search
host = w.config.host
vs_mcp_url = f"{host}/api/2.0/mcp/vector-search/{cfg.catalog}/{cfg.schema}"

logger.info(f"Vector Search MCP URL: {vs_mcp_url}")

# Discover tools exposed by the Vector Search MCP server
tools: list[ToolInfo] = asyncio.run(create_mcp_tools(w=w, url_list=[vs_mcp_url]))

logger.info(f"Discovered {len(tools)} MCP tool(s):")
for tool in tools:
    logger.info(f"  • {tool.name}")
    logger.info(f"    {tool.spec['function']['description']}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — Tool Specifications

# COMMAND ----------

for tool in tools:
    logger.info(f"Tool: {tool.name}")
    logger.info(json.dumps(tool.spec, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5 — Execute a Tool Directly (Smoke Test)

# COMMAND ----------

if tools:
    search_tool = tools[0]
    logger.info(f"Testing tool: {search_tool.name}")

    # Call the vector search tool directly
    result = search_tool.exec_fn(
        query="What are the obligations of high-risk AI system providers?",
        num_results=3,
    )
    logger.info("Tool result (first 500 chars):")
    logger.info(result[:500] if isinstance(result, str) else str(result)[:500])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6 — Initialise EuPolicyAgent

# COMMAND ----------

from eu_policy_agent.agent import EuPolicyAgent

agent = EuPolicyAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    catalog=cfg.catalog,
    schema=cfg.schema,
    genie_space_id=cfg.genie_space_id or None,
    lakebase_project_id=None,  # No memory for this demo
)

logger.info("EuPolicyAgent created.")
logger.info(f"Active tools: {list(agent._tools_dict.keys())}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7 — Single-Turn Query

# COMMAND ----------

mlflow.set_experiment(cfg.experiment_path or "/Shared/eu-policy-agent-dev")

request = ResponsesAgentRequest(
    input=[
        {
            "role": "user",
            "content": (
                "What are the main obligations for providers of high-risk AI systems "
                "under the EU AI Act?"
            ),
        }
    ]
)

response = agent.predict(request)
logger.info("=" * 80)
logger.info("Agent response:")
logger.info(response.output[-1].content if response.output else "(no output)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8 — More EU Legislation Questions

# COMMAND ----------

sample_questions = [
    "What personal data rights does GDPR grant to data subjects?",
    "What are the key obligations for very large online platforms under the DSA?",
    "How does the EU AI Act classify AI systems by risk level?",
    "What are the cybersecurity incident reporting requirements under NIS2?",
]

for question in sample_questions:
    req = ResponsesAgentRequest(input=[{"role": "user", "content": question}])
    resp = agent.predict(req)
    answer = resp.output[-1].content if resp.output else "(no output)"

    logger.info(f"Q: {question}")
    logger.info(f"A: {answer[:300]}...")
    logger.info("-" * 60)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9 — Tool Design Best Practices
# MAGIC
# MAGIC **Do:**
# MAGIC - Write clear, specific descriptions so the LLM knows when to call the tool
# MAGIC - Use strong types and enum constraints in parameter schemas
# MAGIC - Return structured data (JSON or plain text) — not raw binary
# MAGIC - Keep tools focused: one concern per tool
# MAGIC
# MAGIC **Avoid:**
# MAGIC - Overlapping tool descriptions (the LLM will call the wrong one)
# MAGIC - Slow tools without caching (latency compounds in multi-step loops)
# MAGIC - Tools that mutate state without clear confirmation semantics

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10 — MCP vs Custom Functions
# MAGIC
# MAGIC | | MCP (Databricks managed) | Custom Python function |
# MAGIC |---|---|---|
# MAGIC | Setup | URL + auth only | Manual spec + logic |
# MAGIC | Reusability | High (shared across agents) | Low |
# MAGIC | Custom logic | Limited | Unlimited |
# MAGIC | Best for | Standard retrieval / analytics | Domain-specific operations |
# MAGIC
# MAGIC **Rule of thumb:** use MCP for standard Databricks retrieval (Vector Search,
# MAGIC Genie) and custom functions for any bespoke business logic.
