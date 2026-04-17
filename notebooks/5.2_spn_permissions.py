# Databricks notebook source
"""Week 5 — Grant Service Principal Permissions for Deployment.

Before ``agents.deploy()`` can succeed the deploying SPN must already have
access to every resource declared in the model artefact.  This notebook grants
the minimum required permissions so that the Model Serving service identity can
reach:

  - The Genie Space (CAN_RUN)
  - The Vector Search endpoint (CAN_USE)
  - The SQL Warehouse used by Genie (CAN_USE)

Run this once per environment (dev / acc / prd) using the admin SPN that owns
the workspace resources, or as a workspace admin user.

Prerequisites:
  - Secret ``client_id`` in the ``{env}_SPN`` secret scope.
  - The SPN must already exist in the Databricks account.
"""

# COMMAND ----------

import os

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel
from loguru import logger
from pyspark.sql import SparkSession

from eu_policy_agent.config import get_env, load_config

# COMMAND ----------

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    from dotenv import load_dotenv

    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)
w = WorkspaceClient()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — Resolve SPN Application ID

# COMMAND ----------

# Retrieve the SPN client ID from the secret scope named after the environment.
# In the course workspace the scopes are named dev_SPN, acc_SPN, prd_SPN.
from databricks.sdk.runtime import dbutils  # noqa: E402

spn_app_id = dbutils.secrets.get(f"{env}_SPN", "client_id")
logger.info(f"Granting permissions to SPN: {spn_app_id} (env={env})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — Genie Space: CAN_RUN

# COMMAND ----------

if cfg.genie_space_id:
    w.permissions.update(
        request_object_type="genie",
        request_object_id=cfg.genie_space_id,
        access_control_list=[
            AccessControlRequest(
                service_principal_name=spn_app_id,
                permission_level=PermissionLevel.CAN_RUN,
            )
        ],
    )
    logger.info(f"✓ Genie Space {cfg.genie_space_id}: CAN_RUN granted")
else:
    logger.warning("genie_space_id is not configured — skipping Genie permission.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — Vector Search Endpoint: CAN_USE

# COMMAND ----------

vs_endpoint = w.vector_search_endpoints.get_endpoint(cfg.vector_search_endpoint)
w.permissions.update(
    request_object_type="vector-search-endpoints",
    request_object_id=vs_endpoint.id,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_USE,
        )
    ],
)
logger.info(f"✓ Vector Search endpoint {cfg.vector_search_endpoint}: CAN_USE granted")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — SQL Warehouse: CAN_USE

# COMMAND ----------

w.permissions.update(
    request_object_type="warehouses",
    request_object_id=cfg.warehouse_id,
    access_control_list=[
        AccessControlRequest(
            service_principal_name=spn_app_id,
            permission_level=PermissionLevel.CAN_USE,
        )
    ],
)
logger.info(f"✓ SQL Warehouse {cfg.warehouse_id}: CAN_USE granted")

logger.info("All permissions granted successfully.")
