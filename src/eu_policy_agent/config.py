"""Configuration management for EU Policy Agent."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession


class ProjectConfig(BaseModel):
    """Project configuration model."""

    catalog: str = Field(..., description="Unity Catalog name")
    db_schema: str = Field(..., description="Schema name", alias="schema")
    volume: str = Field(..., description="Volume name")
    llm_endpoint: str = Field(default="", description="LLM endpoint name")
    embedding_endpoint: str = Field(default="", description="Embedding endpoint name")
    warehouse_id: str = Field(default="", description="Warehouse ID")
    vector_search_endpoint: str = Field(
        default="", description="Vector search endpoint name"
    )
    genie_space_id: str | None = Field(
        default=None, description="Genie Space ID for metadata exploration"
    )
    lakebase_project_id: str | None = Field(
        default=None, description="Lakebase project ID for session memory"
    )
    usage_policy_id: str | None = Field(
        default=None,
        description="Databricks serverless usage policy ID for cost attribution",
    )
    experiment_path: str = Field(
        default="",
        description="MLflow experiment path (e.g. /Shared/eu-policy-agent-dev)",
    )
    system_prompt: str = Field(
        default=(
            "You are a helpful AI assistant specialising in EU digital "
            "legislation. Use the provided context to answer questions "
            "accurately, citing specific regulations when possible."
        ),
        description="System prompt for the agent",
    )

    model_config = {"populate_by_name": True}

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file.
            env: Environment name (dev, acc, prd).

        Returns:
            ProjectConfig instance for the requested environment.

        Raises:
            ValueError: If ``env`` is not a valid environment or is missing from
                the config file.
        """
        if env not in ("prd", "acc", "dev"):
            raise ValueError(
                f"Invalid environment: {env!r}. Expected one of 'dev', 'acc', 'prd'."
            )

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if env not in config_data:
            raise ValueError(
                f"Environment {env!r} not found in config file {config_path!r}."
            )

        return cls(**config_data[env])

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def schema(self) -> str:
        """Alias for ``db_schema`` — keeps call-sites concise."""
        return self.db_schema

    @property
    def experiment_name(self) -> str:
        """Alias for ``experiment_path`` used by deployment scripts."""
        return self.experiment_path

    @property
    def full_schema_name(self) -> str:
        """Fully qualified schema name: ``{catalog}.{schema}``."""
        return f"{self.catalog}.{self.db_schema}"

    @property
    def full_volume_path(self) -> str:
        """Filesystem path to the Unity Catalog volume."""
        return f"/Volumes/{self.catalog}/{self.db_schema}/{self.volume}"


def load_config(
    config_path: str = "project_config.yml", env: str = "dev"
) -> ProjectConfig:
    """Load project configuration, searching parent directories if needed.

    Walks up to three levels of parent directories to find ``config_path``
    when a relative path is given — mirrors the notebook working-directory
    behaviour on Databricks.

    Args:
        config_path: Path to the YAML configuration file.
        env: Environment name (dev, acc, prd).

    Returns:
        Resolved ``ProjectConfig`` instance.
    """
    if not Path(config_path).is_absolute():
        current = Path.cwd()
        for _ in range(3):
            candidate = current / config_path
            if candidate.exists():
                config_path = str(candidate)
                break
            current = current.parent

    return ProjectConfig.from_yaml(config_path, env)


def get_env(spark: SparkSession) -> str:
    """Read the ``env`` widget value, falling back to ``"dev"``.

    Used by Databricks notebook tasks and Lakeflow jobs to resolve the
    target environment from the ``env`` base parameter.

    Args:
        spark: Active ``SparkSession``.

    Returns:
        Environment name — one of ``"dev"``, ``"acc"``, ``"prd"``.
    """
    try:
        dbutils = DBUtils(spark)
        return dbutils.widgets.get("env")
    except Exception:
        return "dev"
