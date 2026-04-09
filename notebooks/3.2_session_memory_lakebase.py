# Databricks notebook source
"""
Week 3 — Session Memory with Lakebase
=======================================
Topics covered:
  - What Lakebase is and why it matters for multi-turn agents
  - Provisioning a Lakebase project and the session_messages table
  - LakebaseMemory: save_messages / load_messages
  - Multi-turn conversation demo with EuPolicyAgent + memory
  - Authentication: user (dev) vs service principal (production)

Prerequisites:
  - Lakebase is enabled in your Databricks workspace
  - Fill in `lakebase_project_id` in project_config.yml before running
"""

# COMMAND ----------

import json
import os
import urllib.parse
from uuid import uuid4

import mlflow
import psycopg
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import (
    PostgresAPI,
    Project,
    ProjectDefaultEndpointSettings,
    ProjectSpec,
)
from dotenv import load_dotenv
from google.protobuf.duration_pb2 import Duration
from loguru import logger
from mlflow.types.responses import ResponsesAgentRequest
from pyspark.sql import SparkSession

from eu_policy_agent.config import get_env, load_config
from eu_policy_agent.memory import LakebaseMemory

# COMMAND ----------
# Environment setup

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

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — Why Session Memory?
# MAGIC
# MAGIC Without memory each turn of a conversation starts from scratch.  A user
# MAGIC asking "What about Article 9?" has no context — the agent cannot know
# MAGIC which regulation was being discussed.
# MAGIC
# MAGIC **Lakebase** is a Databricks-managed PostgreSQL database with low latency
# MAGIC and native workspace authentication.  It is ideal for storing conversation
# MAGIC history because:
# MAGIC - Sub-millisecond read latency for small session records
# MAGIC - Native integration with Databricks IAM (no external DB to manage)
# MAGIC - JSONB column type stores arbitrary message dicts without schema changes

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — Provision a Lakebase Project
# MAGIC
# MAGIC A Lakebase project maps to a PostgreSQL cluster.  The project ID should
# MAGIC match `lakebase_project_id` in `project_config.yml`.

# COMMAND ----------

pg_api = PostgresAPI(w.api_client)
project_id = cfg.lakebase_project_id

if not project_id:
    raise ValueError(
        "lakebase_project_id is not set in project_config.yml. "
        "Create a Lakebase project in the Databricks UI, then fill in the ID."
    )

# Create the project if it does not already exist
try:
    project = pg_api.get_project(name=f"projects/{project_id}")
    logger.info(f"✓ Lakebase project already exists: {project_id}")
except Exception:
    logger.info(f"Creating Lakebase project: {project_id}…")
    project = pg_api.create_project(
        project_id=project_id,
        project=Project(
            spec=ProjectSpec(
                display_name=f"eu-policy-agent-{env}",
                default_endpoint_settings=ProjectDefaultEndpointSettings(
                    autoscaling_limit_min_cu=1,
                    autoscaling_limit_max_cu=4,
                    suspend_timeout_duration=Duration(seconds=300),
                ),
            ),
        ),
    ).wait()
    logger.info(f"✓ Lakebase project created: {project.name}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — Connect and Create the session_messages Table

# COMMAND ----------

# Resolve host and generate a short-lived credential (user token in notebooks)
default_branch = next(iter(pg_api.list_branches(parent=project.name)))
endpoint = next(iter(pg_api.list_endpoints(parent=default_branch.name)))
pg_host = endpoint.status.hosts.host
credential = pg_api.generate_database_credential(endpoint=endpoint.name)

user = w.current_user.me()
username = urllib.parse.quote_plus(user.user_name)

conn_string = (
    f"postgresql://{username}:{credential.token}@{pg_host}:5432/"
    "databricks_postgres?sslmode=require"
)

logger.info(f"Connected to Lakebase host: {pg_host}")

# COMMAND ----------

with psycopg.connect(conn_string) as conn:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS session_messages (
            id           SERIAL PRIMARY KEY,
            session_id   TEXT      NOT NULL,
            message_data JSONB     NOT NULL,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_session_messages_session_id
        ON session_messages (session_id)
    """)

logger.info("✓ session_messages table ready")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — Manual Save / Load Test

# COMMAND ----------

test_session_id = f"manual-test-{uuid4()}"
test_messages = [
    {"role": "user", "content": "What are the key provisions of the EU AI Act?"},
    {
        "role": "assistant",
        "content": "The EU AI Act introduces a risk-based framework...",
    },
    {"role": "user", "content": "What counts as a high-risk system?"},
]

with psycopg.connect(conn_string) as conn:
    for msg in test_messages:
        conn.execute(
            "INSERT INTO session_messages (session_id, message_data) VALUES (%s, %s)",
            (test_session_id, json.dumps(msg)),
        )

logger.info(f"Saved {len(test_messages)} messages to session: {test_session_id}")

# COMMAND ----------

with psycopg.connect(conn_string) as conn:
    rows = conn.execute(
        """
        SELECT message_data, created_at
        FROM session_messages
        WHERE session_id = %s
        ORDER BY created_at ASC
        """,
        (test_session_id,),
    ).fetchall()

logger.info(f"Loaded {len(rows)} messages:")
for row in rows:
    logger.info(f"  [{row[1]}] {row[0]}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5 — LakebaseMemory Class

# COMMAND ----------

memory = LakebaseMemory(project_id=project_id)

session_id = f"lakebase-memory-test-{uuid4()}"

messages_to_save = [
    {
        "role": "user",
        "content": "Explain the GDPR's lawful basis for data processing.",
    },
    {
        "role": "assistant",
        "content": (
            "Under GDPR Article 6 there are six lawful bases: consent, contract, "
            "legal obligation, vital interests, public task, and legitimate interests."
        ),
    },
]

memory.save_messages(session_id, messages_to_save)
logger.info(f"✓ Saved messages to session: {session_id}")

# COMMAND ----------

loaded = memory.load_messages(session_id)
logger.info(f"Loaded {len(loaded)} messages:")
for msg in loaded:
    logger.info(f"  {msg['role']}: {str(msg['content'])[:80]}…")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6 — Multi-Turn Agent Conversation with Memory

# COMMAND ----------

from eu_policy_agent.agent import EuPolicyAgent

agent = EuPolicyAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    catalog=cfg.catalog,
    schema=cfg.schema,
    genie_space_id=cfg.genie_space_id or None,
    lakebase_project_id=project_id,
)

conversation_session = f"conv-{uuid4().hex[:8]}"
logger.info(f"Starting conversation session: {conversation_session}")

# COMMAND ----------

# Turn 1: ask about the AI Act
turn1 = ResponsesAgentRequest(
    input=[
        {
            "role": "user",
            "content": "What is the EU AI Act and when does it apply?",
        }
    ],
    custom_inputs={"session_id": conversation_session, "request_id": "req-001"},
)

resp1 = agent.predict(turn1)
answer1 = resp1.output[-1].content if resp1.output else "(no output)"
logger.info("Turn 1 — User: What is the EU AI Act and when does it apply?")
logger.info(f"Turn 1 — Agent: {answer1[:400]}…")

# COMMAND ----------

# Turn 2: follow-up that relies on the prior context in memory
turn2 = ResponsesAgentRequest(
    input=[
        {
            "role": "user",
            "content": "What specific obligations does it place on high-risk system providers?",
        }
    ],
    custom_inputs={"session_id": conversation_session, "request_id": "req-002"},
)

resp2 = agent.predict(turn2)
answer2 = resp2.output[-1].content if resp2.output else "(no output)"
logger.info(
    "Turn 2 — User: What specific obligations does it place on high-risk system providers?"
)
logger.info(f"Turn 2 — Agent: {answer2[:400]}…")

# COMMAND ----------

# Verify memory persisted both turns
stored = memory.load_messages(conversation_session)
logger.info(
    f"Messages stored in Lakebase for session {conversation_session!r}: {len(stored)}"
)
for msg in stored:
    logger.info(f"  {msg.get('role')}: {str(msg.get('content', ''))[:60]}…")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7 — Authentication in Production
# MAGIC
# MAGIC In notebooks we authenticate as the current user.  When the agent runs on
# MAGIC **Model Serving**, there is no user token — we must use a **Service Principal**.
# MAGIC
# MAGIC Set these environment variables in your Model Serving endpoint:
# MAGIC ```
# MAGIC LAKEBASE_SP_CLIENT_ID      = <service-principal-client-id>
# MAGIC LAKEBASE_SP_CLIENT_SECRET  = <service-principal-secret>
# MAGIC LAKEBASE_SP_HOST           = https://<workspace>.azuredatabricks.net
# MAGIC ```
# MAGIC
# MAGIC ``LakebaseMemory._get_connection_string()`` detects these env vars
# MAGIC automatically — no code changes needed between dev and production.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8 — Memory Strategy Recommendations
# MAGIC
# MAGIC | Strategy | When to use |
# MAGIC |---|---|
# MAGIC | No memory (stateless) | Simple QA, single-turn use cases |
# MAGIC | Full message history (Lakebase) | Multi-turn conversations, helpdesk |
# MAGIC | Summarisation | Long conversations where context window is a concern |
# MAGIC
# MAGIC **For EU legislation QA**, full history is recommended because follow-up
# MAGIC questions ("What about Article 9?", "Does that apply to SMEs?") rely
# MAGIC heavily on prior context.
