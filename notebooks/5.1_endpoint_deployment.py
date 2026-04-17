# Databricks notebook source
"""Week 5 — Agent Deployment & Endpoint Testing.

Topics covered:
  - Deploying agents with ``agents.deploy()``
  - Configuring environment variables and Lakebase SPN secrets
  - Passing the MLflow experiment ID to route production traces correctly
  - Workload sizing and scale-to-zero
  - Testing the deployed endpoint via the OpenAI Responses API client

Prerequisites:
  - A registered model with the ``latest-model`` alias set
    (see notebook 4.3 / the ``log_register_agent`` job)
  - A secret scope ``eu-policy-agent-scope`` with ``client-id``,
    ``client-secret`` values for the Lakebase SPN
  - For local execution: ``pip install mlflow[databricks]``
"""

# COMMAND ----------

import os
import random
from datetime import datetime

import mlflow
from databricks import agents
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow import MlflowClient
from pyspark.sql import SparkSession

from eu_policy_agent.config import get_env, load_config

# COMMAND ----------
# Environment / MLflow setup

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    from dotenv import load_dotenv

    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

model_name = f"{cfg.catalog}.{cfg.schema}.eu_policy_agent"
endpoint_name = f"eu-policy-agent-endpoint-{env}"
secret_scope = "eu-policy-agent-scope"

client = MlflowClient()
model_version = client.get_model_version_by_alias(model_name, "latest-model").version

workspace = WorkspaceClient()
experiment = client.get_experiment_by_name(cfg.experiment_name)

logger.info(f"Model   : {model_name}  (version {model_version})")
logger.info(f"Endpoint: {endpoint_name}")
logger.info(f"Env     : {env}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — Deploy Agent
# MAGIC
# MAGIC ``agents.deploy()`` handles:
# MAGIC - Endpoint creation and version routing
# MAGIC - Service-identity provisioning for declared resources
# MAGIC - Review app and real-time trace propagation
# MAGIC - Inference tables (payload logging)
# MAGIC
# MAGIC **Important:** pass ``MLFLOW_EXPERIMENT_ID`` so production traces land
# MAGIC in the same experiment used for offline evaluation — not in a new one
# MAGIC created automatically by the deploy command.

# COMMAND ----------

git_sha = "local"

agents.deploy(
    model_name=model_name,
    model_version=int(model_version),
    endpoint_name=endpoint_name,
    usage_policy_id=cfg.usage_policy_id,
    scale_to_zero=True,
    workload_size="Small",
    deploy_feedback_model=False,
    environment_vars={
        "GIT_SHA": git_sha,
        "MODEL_VERSION": model_version,
        "MODEL_SERVING_ENDPOINT_NAME": endpoint_name,
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
        # Lakebase credentials via env vars — declarative resource auth for
        # Lakebase is not yet stable in Model Serving (see week 5 session notes).
        # Do NOT use DATABRICKS_CLIENT_* here; that overrides MCP resource auth.
        "LAKEBASE_SP_CLIENT_ID": f"{{{{secrets/{secret_scope}/client-id}}}}",
        "LAKEBASE_SP_CLIENT_SECRET": f"{{{{secrets/{secret_scope}/client-secret}}}}",
        "LAKEBASE_SP_HOST": workspace.config.host,
    },
)

logger.info("✓ Deployment initiated — endpoint will be ready in ~5-10 minutes.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — Test the Deployed Endpoint
# MAGIC
# MAGIC Wait for the endpoint to reach ``Ready`` state, then run a test query.

# COMMAND ----------

from openai import OpenAI

host = workspace.config.host
token = workspace.tokens.create(lifetime_seconds=2000).token_value

client_openai = OpenAI(
    api_key=token,
    base_url=f"{host}/serving-endpoints",
)

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{ts}-{random.randint(100000, 999999)}"
request_id = f"req-{ts}-{random.randint(100000, 999999)}"

response = client_openai.responses.create(
    model=endpoint_name,
    input=[
        {
            "role": "user",
            "content": (
                "What are the main obligations for providers of high-risk AI systems "
                "under the EU AI Act?"
            ),
        }
    ],
    extra_body={
        "custom_inputs": {
            "session_id": session_id,
            "request_id": request_id,
        }
    },
)

logger.info(f"Response ID : {response.id}")
logger.info(f"Session ID  : {response.custom_outputs.get('session_id')}")
logger.info(f"Request ID  : {response.custom_outputs.get('request_id')}")
logger.info("\nAgent response:")
logger.info("-" * 80)
logger.info(response.output[0].content[0].text)
logger.info("-" * 80)
