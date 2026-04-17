# Databricks notebook source
"""
Week 3 — Genie Space Setup and Integration
===========================================
Topics covered:
  - What is Databricks Genie and when to use it in an agent
  - Creating (or reusing) a Genie Space via the SDK
  - Configuring the space with EU policy tables and column hints
  - Starting and continuing Genie conversations
  - Genie as an MCP tool inside EuPolicyAgent
  - Updating project_config.yml with the Genie Space ID

**What is Genie?**
Databricks Genie is a managed AI data analyst that converts natural language
questions into SQL, executes them against configured Delta tables, and returns
results. Integrated into the EU Policy Agent via MCP, it enables analytics
queries such as:
  - "How many chunks are there per regulation?"
  - "Which regulation has the most articles?"
  - "What is the most recently ingested document?"

**Prerequisites:**
  - Weeks 1–2 complete (eu_policy_chunks table populated)
  - A SQL warehouse configured in project_config.yml (warehouse_id)
  - Genie enabled in your Databricks workspace

**After running this notebook:**
  - Copy the Genie Space ID printed at the end
  - Paste it into genie_space_id in project_config.yml for dev/acc/prd
"""

# COMMAND ----------

import json
import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql
from databricks.sdk.service.sql import CreateWarehouseRequestWarehouseType
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession

from eu_policy_agent.config import get_env, load_config

# COMMAND ----------
# Environment setup

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ["PROFILE"]

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

catalog = cfg.catalog
schema = cfg.schema

w = WorkspaceClient()
logger.info(f"Environment   : {env}")
logger.info(f"Catalog/schema: {catalog}.{schema}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — Check for Existing Genie Space
# MAGIC
# MAGIC If `genie_space_id` is already set in `project_config.yml` this notebook
# MAGIC will skip creation and use the existing space.  You can use it to verify
# MAGIC the space configuration and run test queries.

# COMMAND ----------

if cfg.genie_space_id:
    logger.info(f"Existing Genie Space found in config: {cfg.genie_space_id}")
    space_id = cfg.genie_space_id
    USE_EXISTING_SPACE = True
else:
    logger.info("No Genie Space configured — will create a new one.")
    USE_EXISTING_SPACE = False

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — Warehouse (required by Genie)
# MAGIC
# MAGIC Genie needs a SQL warehouse to execute the generated queries.
# MAGIC We reuse the warehouse already configured in `project_config.yml`
# MAGIC (`warehouse_id`).  If you don't have one yet, the block below creates
# MAGIC a minimal 2X-Small serverless warehouse.

# COMMAND ----------

if cfg.warehouse_id:
    warehouse_id = cfg.warehouse_id
    logger.info(f"Using warehouse from config: {warehouse_id}")
else:
    logger.info("No warehouse_id in config — creating a new 2X-Small warehouse…")
    created = w.warehouses.create(
        name="eu-policy-genie-warehouse",
        cluster_size="2X-Small",
        max_num_clusters=1,
        auto_stop_mins=10,
        warehouse_type=CreateWarehouseRequestWarehouseType("PRO"),
        enable_serverless_compute=True,
        tags=sql.EndpointTags(
            custom_tags=[sql.EndpointTagPair(key="Project", value="eu-policy-agent")]
        ),
    ).result()
    warehouse_id = created.id
    logger.info(f"✓ Created warehouse: {warehouse_id}")
    logger.info("  → Add this to warehouse_id in project_config.yml")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — Genie Space Configuration
# MAGIC
# MAGIC The space configuration tells Genie which tables it can query and
# MAGIC provides column hints that improve SQL generation quality.
# MAGIC
# MAGIC We expose two tables:
# MAGIC - **eu_policy_chunks** — the primary text-and-metadata table
# MAGIC - **raw_documents** — source PDF metadata (filename, page count, etc.)

# COMMAND ----------

serialized_space = {
    "version": 1,
    "data_sources": {
        "tables": [
            {
                "identifier": f"{catalog}.{schema}.eu_policy_chunks",
                "column_configs": [
                    {"column_name": "chunk_id", "get_example_values": True},
                    {
                        "column_name": "regulation",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {"column_name": "article", "get_example_values": True},
                    {"column_name": "chunk_text", "get_example_values": True},
                    {"column_name": "chunk_index"},
                    {"column_name": "source_document"},
                ],
            },
            {
                "identifier": f"{catalog}.{schema}.raw_documents",
                "column_configs": [
                    {
                        "column_name": "filename",
                        "get_example_values": True,
                        "build_value_dictionary": True,
                    },
                    {"column_name": "page_count"},
                    {"column_name": "ingested_at"},
                    {"column_name": "volume_path"},
                ],
            },
        ]
    },
}

logger.info("Genie Space configuration:")
logger.info(json.dumps(serialized_space, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — Create or Verify Genie Space

# COMMAND ----------

if not USE_EXISTING_SPACE:
    logger.info("Creating Genie Space…")
    space = w.genie.create_space(
        warehouse_id=warehouse_id,
        serialized_space=json.dumps(serialized_space),
        title="eu-policy-agent-space",
    )
    space_id = space.space_id
    logger.info(f"✓ Created Genie Space: {space_id}")
    logger.info("")
    logger.info("=" * 60)
    logger.info("ACTION REQUIRED: add this to project_config.yml")
    logger.info(f'  genie_space_id: "{space_id}"')
    logger.info("=" * 60)
else:
    logger.info(f"Using existing Genie Space: {space_id}")

# Verify the space is accessible
space_info = w.genie.get_space(space_id=space_id, include_serialized_space=True)
logger.info("✓ Genie Space verified")
logger.info(f"  Space ID: {space_id}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5 — Test: Start a Conversation
# MAGIC
# MAGIC Ask Genie a natural language question about the EU policy data.
# MAGIC This verifies the space is correctly wired to the Delta tables.

# COMMAND ----------

logger.info("Starting Genie conversation…")
conversation = w.genie.start_conversation_and_wait(
    space_id=space_id,
    content="How many chunks are there per regulation in the eu_policy_chunks table?",
)

logger.info("Genie response:")
logger.info(json.dumps(conversation.as_dict(), indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6 — Continue the Conversation (Multi-Turn)

# COMMAND ----------

follow_up = w.genie.create_message_and_wait(
    space_id=space_id,
    conversation_id=conversation.conversation_id,
    content="Which regulation has the highest average chunk length?",
)

logger.info("Follow-up response:")
logger.info(json.dumps(follow_up.as_dict(), indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7 — Genie as an MCP Tool in EuPolicyAgent
# MAGIC
# MAGIC Once `genie_space_id` is set in `project_config.yml`, `EuPolicyAgent`
# MAGIC automatically includes the Genie MCP tool alongside Vector Search.
# MAGIC
# MAGIC The agent will call Genie when the question is analytical in nature
# MAGIC (counting, aggregating, filtering by metadata) and Vector Search when
# MAGIC semantic similarity is needed (finding relevant legislative text).
# MAGIC
# MAGIC ```python
# MAGIC # With genie_space_id set, the agent gets both tools:
# MAGIC agent = EuPolicyAgent(
# MAGIC     llm_endpoint=cfg.llm_endpoint,
# MAGIC     system_prompt=cfg.system_prompt,
# MAGIC     catalog=cfg.catalog,
# MAGIC     schema=cfg.schema,
# MAGIC     genie_space_id=cfg.genie_space_id,  # ← enables Genie MCP tool
# MAGIC     lakebase_project_id=cfg.lakebase_project_id,
# MAGIC )
# MAGIC # agent._tools_dict now contains:
# MAGIC #   dev__eu_policy__eu_policy_index  (Vector Search)
# MAGIC #   ask_genie                         (Genie Space)
# MAGIC ```
# MAGIC
# MAGIC → See `3.1_agent_tool_calling.py` for a full agent demo with both tools.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8 — Summary and Next Steps
# MAGIC
# MAGIC **What was configured:**
# MAGIC - Genie Space connected to `eu_policy_chunks` and `raw_documents`
# MAGIC - Natural language → SQL tested with multi-turn conversation
# MAGIC - Space ID ready for `project_config.yml`
# MAGIC
# MAGIC **Next steps:**
# MAGIC 1. Copy the Space ID above and set `genie_space_id` in `project_config.yml`
# MAGIC    for each environment (dev/acc/prd).
# MAGIC 2. Re-run `3.1_agent_tool_calling.py` — the agent will now have both
# MAGIC    Vector Search and Genie tools available.
# MAGIC 3. When logging the model (notebook `4.3_mlflow_log_register.py`), the
# MAGIC    `DatabricksGenieSpace` resource will be automatically declared.
# MAGIC
# MAGIC **Production note:** In Model Serving, the agent's service identity must
# MAGIC have SELECT permission on the Genie-managed tables and access to the SQL
# MAGIC warehouse. These permissions are granted automatically when you declare
# MAGIC `DatabricksGenieSpace` in the model resources.

# COMMAND ----------

logger.info("=" * 60)
logger.info("Genie Space setup complete.")
logger.info(f"Space ID: {space_id}")
logger.info("")
logger.info("To enable in the agent, add to project_config.yml:")
logger.info(f'  genie_space_id: "{space_id}"')
logger.info("=" * 60)
