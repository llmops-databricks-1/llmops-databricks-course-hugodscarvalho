"""Unit tests for eu_policy_agent.config.

Tests cover:
- ProjectConfig pydantic model: valid construction, missing required fields,
  alias resolution, and default values.
- Environment validation in ProjectConfig.from_yaml.
- YAML loading via from_yaml - happy path and error branches.
- Computed properties: schema, full_schema_name, full_volume_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eu_policy_agent.config import ProjectConfig

# ProjectConfig - model construction


class TestProjectConfigConstruction:
    """ProjectConfig can be constructed from keyword arguments."""

    def test_minimal_required_fields(self) -> None:
        """catalog, schema, and volume are the only truly required fields."""
        cfg = ProjectConfig(catalog="cat", schema="sch", volume="vol")
        assert cfg.catalog == "cat"
        assert cfg.db_schema == "sch"
        assert cfg.volume == "vol"

    def test_all_fields(self) -> None:
        """All optional fields are accepted and stored correctly."""
        cfg = ProjectConfig(
            catalog="cat",
            schema="sch",
            volume="vol",
            llm_endpoint="llm-ep",
            embedding_endpoint="emb-ep",
            warehouse_id="wh-123",
            vector_search_endpoint="vs-ep",
            genie_space_id="genie-abc",
        )
        assert cfg.llm_endpoint == "llm-ep"
        assert cfg.embedding_endpoint == "emb-ep"
        assert cfg.warehouse_id == "wh-123"
        assert cfg.vector_search_endpoint == "vs-ep"
        assert cfg.genie_space_id == "genie-abc"

    def test_optional_fields_default_to_empty_string(self) -> None:
        """Optional string fields default to empty string, not None."""
        cfg = ProjectConfig(catalog="cat", schema="sch", volume="vol")
        assert cfg.llm_endpoint == ""
        assert cfg.embedding_endpoint == ""
        assert cfg.warehouse_id == ""
        assert cfg.vector_search_endpoint == ""

    def test_genie_space_id_defaults_to_none(self) -> None:
        cfg = ProjectConfig(catalog="cat", schema="sch", volume="vol")
        assert cfg.genie_space_id is None

    def test_schema_alias_populates_db_schema(self) -> None:
        """The pydantic alias 'schema' writes to the db_schema field."""
        cfg = ProjectConfig(catalog="c", schema="my_schema", volume="v")
        assert cfg.db_schema == "my_schema"

    def test_missing_required_field_raises(self) -> None:
        """Omitting a required field must raise a pydantic ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ProjectConfig(schema="sch", volume="vol")  # missing catalog

    def test_system_prompt_has_sensible_default(self) -> None:
        cfg = ProjectConfig(catalog="c", schema="s", volume="v")
        assert "EU" in cfg.system_prompt
        assert len(cfg.system_prompt) > 20


# ProjectConfig - computed properties


class TestProjectConfigProperties:
    """Computed properties derive the correct strings from model fields."""

    def test_schema_property_mirrors_db_schema(
        self, sample_config: ProjectConfig
    ) -> None:
        assert sample_config.schema == sample_config.db_schema

    def test_full_schema_name(self, sample_config: ProjectConfig) -> None:
        expected = f"{sample_config.catalog}.{sample_config.db_schema}"
        assert sample_config.full_schema_name == expected

    def test_full_volume_path(self, sample_config: ProjectConfig) -> None:
        cfg = ProjectConfig(catalog="my_cat", schema="my_sch", volume="my_vol")
        assert cfg.full_volume_path == "/Volumes/my_cat/my_sch/my_vol"


# ProjectConfig.from_yaml - environment validation


class TestFromYamlEnvValidation:
    """from_yaml rejects invalid or missing environment names."""

    def test_invalid_env_raises_value_error(self, config_yaml_path: Path) -> None:
        with pytest.raises(ValueError, match="Invalid environment"):
            ProjectConfig.from_yaml(str(config_yaml_path), env="staging")

    @pytest.mark.parametrize("env", ["dev", "acc", "prd"])
    def test_valid_envs_are_accepted(self, config_yaml_path: Path, env: str) -> None:
        cfg = ProjectConfig.from_yaml(str(config_yaml_path), env=env)
        assert cfg.catalog == f"{env}_catalog"
        assert cfg.db_schema == f"{env}_schema"
        assert cfg.volume == f"{env}_volume"

    def test_missing_env_key_in_yaml_raises(self, tmp_path: Path) -> None:
        """A YAML that lacks the requested env key should raise ValueError."""
        sparse_yaml = tmp_path / "sparse.yml"
        sparse_yaml.write_text("dev:\n  catalog: c\n  schema: s\n  volume: v\n")
        with pytest.raises(ValueError, match="not found in config"):
            ProjectConfig.from_yaml(str(sparse_yaml), env="prd")

    def test_nonexistent_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            ProjectConfig.from_yaml("/nonexistent/path/config.yml")


# ProjectConfig.from_yaml - happy-path field hydration


class TestFromYamlFieldHydration:
    """from_yaml correctly populates optional fields from YAML values."""

    def test_optional_fields_loaded_from_yaml(self, config_yaml_path: Path) -> None:
        cfg = ProjectConfig.from_yaml(str(config_yaml_path), env="dev")
        assert cfg.llm_endpoint == "dev-llm"
        assert cfg.embedding_endpoint == "dev-embedding"
        assert cfg.vector_search_endpoint == "dev-vs-endpoint"

    def test_missing_optional_fields_use_defaults(self, config_yaml_path: Path) -> None:
        """acc / prd entries in the fixture YAML have no optional keys."""
        cfg = ProjectConfig.from_yaml(str(config_yaml_path), env="acc")
        assert cfg.llm_endpoint == ""
        assert cfg.embedding_endpoint == ""
