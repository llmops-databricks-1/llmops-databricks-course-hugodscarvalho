"""Shared test configuration and fixtures.

pyspark (via databricks-connect) is a **dev** extra, not a **ci** extra,
so it is absent from the CI environment.  We stub the affected modules at
module-load time - before pytest collects any test file that imports the
package - so that ``import eu_policy_agent`` succeeds without a live cluster.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Module-level stubs - must run before test collection

_PYSPARK_STUBS = (
    "pyspark",
    "pyspark.sql",
    "pyspark.sql.functions",
    "pyspark.sql.types",
    "pyspark.dbutils",
)


def _stub_if_missing(name: str) -> None:
    """Register a MagicMock stub only when the real module cannot be imported."""
    if name in sys.modules:
        return
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = MagicMock()


for _mod in _PYSPARK_STUBS:
    _stub_if_missing(_mod)


# Deferred package import (after stubs are in place)

from eu_policy_agent.config import ProjectConfig  # noqa: E402

# Shared fixtures

_MINIMAL_YAML = """\
dev:
  catalog: dev_catalog
  schema: dev_schema
  volume: dev_volume
  llm_endpoint: dev-llm
  embedding_endpoint: dev-embedding
  vector_search_endpoint: dev-vs-endpoint

acc:
  catalog: acc_catalog
  schema: acc_schema
  volume: acc_volume

prd:
  catalog: prd_catalog
  schema: prd_schema
  volume: prd_volume
"""


@pytest.fixture()
def sample_config() -> ProjectConfig:
    """Minimal valid ProjectConfig for unit tests."""
    return ProjectConfig(
        catalog="test_catalog",
        schema="test_schema",
        volume="test_volume",
        llm_endpoint="test-llm",
        embedding_endpoint="test-embedding",
        vector_search_endpoint="test-vs-endpoint",
    )


@pytest.fixture()
def config_yaml_path(tmp_path: Path) -> Path:
    """Write a minimal project_config.yml to a temp dir and return its path."""
    config_file = tmp_path / "project_config.yml"
    config_file.write_text(_MINIMAL_YAML)
    return config_file


@pytest.fixture()
def mock_spark() -> MagicMock:
    """Return a MagicMock that quacks like a SparkSession."""
    return MagicMock()
