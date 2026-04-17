# Databricks notebook source
"""Week 5 — Deploy the EU Policy Agent to a Model Serving endpoint (CI/CD task).

This script is the second task in the ``register_deploy_agent`` Lakeflow job.
It resolves the ``latest-model`` alias set by the preceding log/register task,
then calls ``agents.deploy()`` which:

  * Creates or updates the model serving endpoint.
  * Provisions service-identity authentication for all declared resources.
  * Enables the review app and real-time trace propagation.
  * Configures inference tables for payload logging.

Parameters (injected via DABs ``base_parameters``):
    env      -- Target environment: dev | acc | prd.
    git_sha  -- Git commit SHA of the code being deployed.
"""

from databricks import agents
from databricks.sdk import WorkspaceClient
from loguru import logger
from mlflow import MlflowClient
from pyspark.sql import SparkSession

from eu_policy_agent.config import get_env, load_config
from eu_policy_agent.utils.common import get_widget

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_widget("env", get_env(spark))
git_sha = get_widget("git_sha", "local")

secret_scope = "eu-policy-agent-scope"

cfg = load_config("../../project_config.yml", env=env)

model_name = f"{cfg.catalog}.{cfg.schema}.eu_policy_agent"
endpoint_name = f"eu-policy-agent-endpoint-{env}"

client = MlflowClient()
model_version = client.get_model_version_by_alias(model_name, "latest-model").version

experiment = client.get_experiment_by_name(cfg.experiment_name)

logger.info("Deploying EU Policy Agent:")
logger.info(f"  Model   : {model_name}")
logger.info(f"  Version : {model_version}")
logger.info(f"  Endpoint: {endpoint_name}")
logger.info(f"  Env     : {env}")

# COMMAND ----------

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
        # Routing traces to the same experiment used during log/register keeps
        # the full lifecycle (offline eval + production traces) in one place.
        "MLFLOW_EXPERIMENT_ID": experiment.experiment_id,
        # Lakebase credentials are passed as env vars because the declarative
        # resource authentication for Lakebase is not yet stable in Model Serving.
        # See session notes week 5 for the full explanation.
        "LAKEBASE_SP_CLIENT_ID": f"{{{{secrets/{secret_scope}/client-id}}}}",
        "LAKEBASE_SP_CLIENT_SECRET": f"{{{{secrets/{secret_scope}/client-secret}}}}",
        "LAKEBASE_SP_HOST": WorkspaceClient().config.host,
    },
)

logger.info("✓ Deployment complete!")
