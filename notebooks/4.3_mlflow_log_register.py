# Databricks notebook source
"""
Week 4 — Log and Register the EU Policy Agent to Unity Catalog
================================================================
Topics covered:
  - Running evaluation before logging (quality gate)
  - mlflow.pyfunc.log_model with ResourceDeclarations
  - Registering the model to Unity Catalog
  - Setting model aliases (champion/challenger pattern)
  - The log_register_agent() helper from eu_policy_agent.agent
  - Resources: why declaring them matters for Model Serving

Resource declarations tell Databricks Model Serving which external services
the agent depends on, so the platform can provision the right service identity
and permissions automatically.

Resources declared for the EU Policy Agent:
  - DatabricksServingEndpoint: LLM and embedding endpoints
  - DatabricksVectorSearchIndex: eu_policy_index
  - DatabricksTable: eu_policy_chunks, raw_documents
  - DatabricksSQLWarehouse: for Genie queries (optional)
"""

# COMMAND ----------

import os
import random
from datetime import datetime

import mlflow
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from loguru import logger
from mlflow import MlflowClient
from mlflow.models.resources import (
    DatabricksGenieSpace,
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksVectorSearchIndex,
)
from pyspark.sql import SparkSession

from eu_policy_agent.agent import EuPolicyAgent
from eu_policy_agent.config import get_env, load_config
from eu_policy_agent.evaluation import (
    cites_regulation_guideline,
    create_eval_data_from_file,
    mentions_legislation,
    polite_tone_guideline,
    stays_in_scope_guideline,
    word_count_check,
)

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

experiment_path = cfg.experiment_path or "/Shared/eu-policy-agent-dev"
mlflow.set_experiment(experiment_path)

git_sha = os.getenv("GIT_SHA", "local")
run_id = os.getenv("RUN_ID", "unset")

logger.info(f"Environment: {env}")
logger.info(f"Git SHA: {git_sha}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — Initialise Agent and Run Evaluation (Quality Gate)

# COMMAND ----------

agent = EuPolicyAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    catalog=cfg.catalog,
    schema=cfg.schema,
    genie_space_id=cfg.genie_space_id or None,
    lakebase_project_id=None,  # Stateless for evaluation run
)


def predict_fn(question: str) -> str:
    """Evaluation predict wrapper."""
    request = {"input": [{"role": "user", "content": question}]}
    result = agent.predict(request)
    if not result.output:
        return ""
    content = result.output[-1].content
    if isinstance(content, list):
        return " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return str(content)


# COMMAND ----------

eval_data = create_eval_data_from_file("../eval_inputs.txt")
logger.info(f"Evaluating on {len(eval_data)} question(s)…")

eval_results = mlflow.genai.evaluate(
    predict_fn=predict_fn,
    data=eval_data,
    scorers=[
        polite_tone_guideline,
        stays_in_scope_guideline,
        cites_regulation_guideline,
        word_count_check,
        mentions_legislation,
    ],
)

logger.info("Evaluation metrics:")
for name, value in eval_results.metrics.items():
    logger.info(
        f"  {name}: {value:.3f}" if isinstance(value, float) else f"  {name}: {value}"
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — Declare Resources
# MAGIC
# MAGIC Resources are required so Model Serving can grant the deployed agent
# MAGIC access to the external services it depends on.

# COMMAND ----------

resources = [
    DatabricksServingEndpoint(endpoint_name=cfg.llm_endpoint),
    DatabricksServingEndpoint(endpoint_name=cfg.embedding_endpoint),
    DatabricksVectorSearchIndex(index_name=f"{cfg.catalog}.{cfg.schema}.eu_policy_index"),
    DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.eu_policy_chunks"),
    DatabricksTable(table_name=f"{cfg.catalog}.{cfg.schema}.raw_documents"),
    DatabricksSQLWarehouse(warehouse_id=cfg.warehouse_id),
]
# Declare Genie Space resource only when configured — Model Serving needs this
# to grant the service identity access to the Genie Space and its SQL warehouse.
if cfg.genie_space_id:
    resources.append(DatabricksGenieSpace(genie_space_id=cfg.genie_space_id))

logger.info(f"Declared {len(resources)} resource(s):")
for r in resources:
    logger.info(f"  {type(r).__name__}: {r}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — Build Model Config and Test Request

# COMMAND ----------

model_config = {
    "catalog": cfg.catalog,
    "schema": cfg.schema,
    "genie_space_id": cfg.genie_space_id or "",
    "system_prompt": cfg.system_prompt,
    "llm_endpoint": cfg.llm_endpoint,
    "lakebase_project_id": cfg.lakebase_project_id or "",
}

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{ts}-{random.randint(100000, 999999)}"
request_id = f"req-{ts}-{random.randint(100000, 999999)}"

test_request = {
    "input": [
        {
            "role": "user",
            "content": (
                "What are the main obligations for providers of high-risk AI systems "
                "under the EU AI Act?"
            ),
        }
    ],
    "custom_inputs": {
        "session_id": session_id,
        "request_id": request_id,
    },
}

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — Log Model to MLflow

# COMMAND ----------

ts_date = datetime.now().strftime("%Y-%m-%d")

with mlflow.start_run(
    run_name=f"eu-policy-agent-{ts_date}",
    tags={"git_sha": git_sha, "run_id": run_id},
) as run:
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model="../agent_serving.py",
        resources=resources,
        input_example=test_request,
        model_config=model_config,
    )
    # Log evaluation metrics alongside the model artefact
    mlflow.log_metrics(eval_results.metrics)

logger.info(f"✓ Model logged: {model_info.model_uri}")
logger.info(f"  Run ID: {run.info.run_id}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5 — Register to Unity Catalog

# COMMAND ----------

model_name = f"{cfg.catalog}.{cfg.schema}.eu_policy_agent"
logger.info(f"Registering model: {model_name}")

registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=model_name,
    env_pack="databricks_model_serving",
    tags={"git_sha": git_sha, "run_id": run_id},
)

logger.info(f"✓ Registered version: {registered_model.version}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6 — Set Model Alias (Champion/Challenger Pattern)

# COMMAND ----------

client = MlflowClient()

# Set the 'champion' alias to the newly registered version
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=registered_model.version,
)
logger.info(f"✓ Alias 'champion' → version {registered_model.version}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7 — Using the log_register_agent() Helper
# MAGIC
# MAGIC The same steps above are encapsulated in
# MAGIC ``eu_policy_agent.agent.log_register_agent()`` for use in automated
# MAGIC pipelines (DABs jobs, CI/CD):

# COMMAND ----------

# Equivalent one-liner (commented out to avoid double-registration):
# registered = log_register_agent(
#     cfg=cfg,
#     git_sha=git_sha,
#     run_id=run_id,
#     agent_code_path="../agent_serving.py",
#     model_name=model_name,
#     evaluation_metrics=eval_results.metrics,
# )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8 — Verify the Registered Model

# COMMAND ----------

registered = client.get_registered_model(model_name)
latest_versions = client.get_latest_versions(model_name)

logger.info(f"Model: {registered.name}")
logger.info(f"Latest versions: {[v.version for v in latest_versions]}")

champion_version = client.get_model_version_by_alias(model_name, "champion")
logger.info(f"Champion alias → version {champion_version.version}")
logger.info(f"  Tags: {champion_version.tags}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9 — Next Steps: Deployment (Week 5)
# MAGIC
# MAGIC The registered model can now be deployed to a **Model Serving endpoint**:
# MAGIC
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.sdk.service.serving import (
# MAGIC     EndpointCoreConfigInput,
# MAGIC     ServedModelInput,
# MAGIC )
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC w.serving_endpoints.create_and_wait(
# MAGIC     name="eu-policy-agent-endpoint",
# MAGIC     config=EndpointCoreConfigInput(
# MAGIC         served_models=[
# MAGIC             ServedModelInput(
# MAGIC                 model_name=model_name,
# MAGIC                 model_version=champion_version.version,
# MAGIC                 scale_to_zero_enabled=True,
# MAGIC             )
# MAGIC         ]
# MAGIC     ),
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC This will be covered in Week 5 (Agent Deployment & Monitoring).
