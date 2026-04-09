"""MLflow model serving entry point for EuPolicyAgent.

This file is referenced by ``mlflow.pyfunc.log_model(python_model=...)``
during model registration.  MLflow will execute it inside the serving
container to instantiate the model.

Configuration is provided via ``ModelConfig`` (populated from the
``model_config`` dict logged alongside the model artefact).

IMPORTANT: This file is intentionally named ``agent_serving.py`` (not
``eu_policy_agent.py``) to avoid shadowing the ``eu_policy_agent`` package
on ``sys.path`` during local development and testing.

Do NOT import this file directly in notebooks or application code.
Use the ``eu_policy_agent`` package (``from eu_policy_agent.agent import
EuPolicyAgent``) instead.
"""

import mlflow
from mlflow.models import ModelConfig

from eu_policy_agent.agent import EuPolicyAgent

config = ModelConfig(
    development_config={
        "catalog": "dev",
        "schema": "eu_policy",
        "genie_space_id": "",
        "system_prompt": (
            "You are a helpful AI assistant specialising in EU digital legislation. "
            "Use the provided context to answer questions accurately, citing specific "
            "regulations and article numbers when possible. "
            "If you cannot find the answer in the retrieved context, say so clearly "
            "rather than guessing."
        ),
        "llm_endpoint": "databricks-llama-4-maverick",
        "lakebase_project_id": "",
    }
)

agent = EuPolicyAgent(
    llm_endpoint=config.get("llm_endpoint"),
    system_prompt=config.get("system_prompt"),
    catalog=config.get("catalog"),
    schema=config.get("schema"),
    genie_space_id=config.get("genie_space_id") or None,
    lakebase_project_id=config.get("lakebase_project_id") or None,
)
mlflow.models.set_model(agent)
