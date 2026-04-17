# Databricks notebook source
"""Week 5 — Log, Evaluate and Register the EU Policy Agent (CI/CD task).

This script is the first task in the ``register_deploy_agent`` Lakeflow job.
It runs the evaluation suite as a quality gate, then logs and registers the
agent model to Unity Catalog, stamping the new version with the ``latest-model``
alias so the downstream deploy task can resolve it without hard-coding a version.

Parameters (injected via DABs ``base_parameters``):
    env      -- Target environment: dev | acc | prd.
    git_sha  -- Git commit SHA of the code being deployed.
    run_id   -- Lakeflow job run ID for cross-system traceability.
"""

import mlflow
from pyspark.sql import SparkSession

from eu_policy_agent.agent import log_register_agent
from eu_policy_agent.config import get_env, load_config
from eu_policy_agent.evaluation import evaluate_agent
from eu_policy_agent.utils.common import get_widget

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Prefer the widget value set by the Lakeflow job; fall back to env var / defaults.
env = get_widget("env", get_env(spark))
git_sha = get_widget("git_sha", "local")
run_id = get_widget("run_id", "local")

cfg = load_config("../../project_config.yml", env=env)

mlflow.set_experiment(cfg.experiment_name)

model_name = f"{cfg.catalog}.{cfg.schema}.eu_policy_agent"

# COMMAND ----------

# Run evaluation — acts as a quality gate before registration.
# Failures here will prevent the model from being registered and
# the deploy task from running.
results = evaluate_agent(cfg, eval_inputs_path="../../eval_inputs.txt")

# COMMAND ----------

# Log, register, and stamp the 'latest-model' alias.
log_register_agent(
    cfg=cfg,
    git_sha=git_sha,
    run_id=run_id,
    agent_code_path="../../agent_serving.py",
    model_name=model_name,
    evaluation_metrics=results.metrics,
)
