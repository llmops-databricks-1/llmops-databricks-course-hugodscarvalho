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

    model_config = {"populate_by_name": True}

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev") -> "ProjectConfig":
        """Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file
            env: Environment name (dev, acc, prd)

        Returns:
            ProjectConfig instance
        """
        if env not in ["prd", "acc", "dev"]:
            raise ValueError(
                f"Invalid environment: {env}. Expected 'prd', 'acc', or 'dev'"
            )

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        if env not in config_data:
            raise ValueError(f"Environment '{env}' not found in config file")

        return cls(**config_data[env])

    @property
    def schema(self) -> str:
        """Alias for db_schema for backward compatibility."""
        return self.db_schema

    @property
    def full_schema_name(self) -> str:
        """Get fully qualified schema name."""
        return f"{self.catalog}.{self.db_schema}"

    @property
    def full_volume_path(self) -> str:
        """Get fully qualified volume path as filesystem path."""
        return f"/Volumes/{self.catalog}/{self.db_schema}/{self.volume}"


def load_config(
    config_path: str = "project_config.yml", env: str = "dev"
) -> ProjectConfig:
    """Load project configuration.

    Args:
        config_path: Path to configuration file
        env: Environment name

    Returns:
        ProjectConfig instance
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
    """Get current environment from dbutils widget, falling back to 'dev'.

    Args:
        spark: Active SparkSession

    Returns:
        Environment name (dev, acc, or prd)
    """
    try:
        dbutils = DBUtils(spark)
        return dbutils.widgets.get("env")
    except Exception:
        return "dev"
