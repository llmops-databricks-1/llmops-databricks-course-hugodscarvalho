"""Unit tests for eu_policy_agent.vector_search.

Strategy
--------
VectorSearchClient makes network calls to a Databricks cluster.
We patch it at module level for every test that instantiates
VectorSearchManager, keeping all assertions on pure Python behaviour:
property values, kwargs construction, and branching logic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from eu_policy_agent.config import ProjectConfig
from eu_policy_agent.vector_search import VectorSearchManager

_PATCH_CLIENT = "eu_policy_agent.vector_search.VectorSearchClient"


# Shared fixture


@pytest.fixture()
def vs_manager(sample_config: ProjectConfig) -> VectorSearchManager:
    """Return a VectorSearchManager with the Databricks client mocked out."""
    with patch(_PATCH_CLIENT):
        mgr = VectorSearchManager(config=sample_config)
    # mgr.client is the MagicMock instance - callers can configure it freely.
    return mgr


# VectorSearchManager.parse_results (static - no client needed)


class TestParseResults:
    """parse_results converts the raw Vector Search API response to dicts."""

    def test_happy_path(self) -> None:
        raw = {
            "manifest": {"columns": [{"name": "id"}, {"name": "text"}]},
            "result": {"data_array": [["doc1_0", "Some text"], ["doc2_1", "More"]]},
        }
        result = VectorSearchManager.parse_results(raw)
        assert result == [
            {"id": "doc1_0", "text": "Some text"},
            {"id": "doc2_1", "text": "More"},
        ]

    def test_empty_data_array_returns_empty_list(self) -> None:
        raw = {
            "manifest": {"columns": [{"name": "id"}]},
            "result": {"data_array": []},
        }
        assert VectorSearchManager.parse_results(raw) == []

    def test_missing_manifest_with_no_data_returns_empty_list(self) -> None:
        # columns resolves to [], data_array is also empty -> no rows to zip
        raw: dict = {"result": {"data_array": []}}
        assert VectorSearchManager.parse_results(raw) == []

    def test_missing_manifest_with_data_raises_value_error(self) -> None:
        # strict=True in zip() raises when column count != row length
        raw: dict = {"result": {"data_array": [["v1"]]}}
        with pytest.raises(ValueError):
            VectorSearchManager.parse_results(raw)

    def test_missing_result_key_returns_empty_list(self) -> None:
        raw: dict = {"manifest": {"columns": [{"name": "id"}]}}
        assert VectorSearchManager.parse_results(raw) == []

    def test_empty_response_returns_empty_list(self) -> None:
        assert VectorSearchManager.parse_results({}) == []

    def test_preserves_column_order(self) -> None:
        raw = {
            "manifest": {"columns": [{"name": "a"}, {"name": "b"}, {"name": "c"}]},
            "result": {"data_array": [[1, 2, 3]]},
        }
        result = VectorSearchManager.parse_results(raw)
        assert list(result[0].keys()) == ["a", "b", "c"]


# VectorSearchManager - initialisation and properties


class TestVectorSearchManagerInit:
    """VectorSearchManager resolves names and stores config on construction."""

    @patch(_PATCH_CLIENT)
    def test_source_table_fqn(self, _: MagicMock, sample_config: ProjectConfig) -> None:
        mgr = VectorSearchManager(config=sample_config)
        assert mgr.source_table == "test_catalog.test_schema.eu_policy_chunks"

    @patch(_PATCH_CLIENT)
    def test_index_name_fqn(self, _: MagicMock, sample_config: ProjectConfig) -> None:
        mgr = VectorSearchManager(config=sample_config)
        assert mgr.index_name == "test_catalog.test_schema.eu_policy_index"

    @patch(_PATCH_CLIENT)
    def test_endpoint_falls_back_to_config(
        self, _: MagicMock, sample_config: ProjectConfig
    ) -> None:
        mgr = VectorSearchManager(config=sample_config)
        assert mgr.endpoint_name == sample_config.vector_search_endpoint

    @patch(_PATCH_CLIENT)
    def test_endpoint_override_takes_precedence(
        self, _: MagicMock, sample_config: ProjectConfig
    ) -> None:
        mgr = VectorSearchManager(config=sample_config, endpoint_name="custom-ep")
        assert mgr.endpoint_name == "custom-ep"

    @patch(_PATCH_CLIENT)
    def test_embedding_falls_back_to_config(
        self, _: MagicMock, sample_config: ProjectConfig
    ) -> None:
        mgr = VectorSearchManager(config=sample_config)
        assert mgr.embedding_model == sample_config.embedding_endpoint

    @patch(_PATCH_CLIENT)
    def test_embedding_override_takes_precedence(
        self, _: MagicMock, sample_config: ProjectConfig
    ) -> None:
        mgr = VectorSearchManager(config=sample_config, embedding_model="custom-emb")
        assert mgr.embedding_model == "custom-emb"


# VectorSearchManager.search - kwargs construction


def _stub_index(manager: VectorSearchManager) -> MagicMock:
    """Configure manager.client.get_index to return a fresh mock index."""
    mock_index = MagicMock()
    mock_index.similarity_search.return_value = {
        "manifest": {"columns": []},
        "result": {"data_array": []},
    }
    manager.client.get_index.return_value = mock_index
    return mock_index


class TestSearch:
    """search correctly assembles kwargs before calling similarity_search."""

    def test_basic_search_passes_query_and_default_num_results(
        self, vs_manager: VectorSearchManager
    ) -> None:
        mock_index = _stub_index(vs_manager)
        vs_manager.search("EU AI Act")
        kwargs = mock_index.similarity_search.call_args.kwargs
        assert kwargs["query_text"] == "EU AI Act"
        assert kwargs["num_results"] == 5

    def test_default_columns_include_key_fields(
        self, vs_manager: VectorSearchManager
    ) -> None:
        mock_index = _stub_index(vs_manager)
        vs_manager.search("GDPR")
        kwargs = mock_index.similarity_search.call_args.kwargs
        for field in ("id", "text", "document_id"):
            assert field in kwargs["columns"]

    def test_custom_num_results(self, vs_manager: VectorSearchManager) -> None:
        mock_index = _stub_index(vs_manager)
        vs_manager.search("query", num_results=10)
        assert mock_index.similarity_search.call_args.kwargs["num_results"] == 10

    def test_filters_included_when_provided(
        self, vs_manager: VectorSearchManager
    ) -> None:
        mock_index = _stub_index(vs_manager)
        vs_manager.search("query", filters={"year": "2024"})
        assert mock_index.similarity_search.call_args.kwargs["filters"] == {
            "year": "2024"
        }

    def test_filters_omitted_when_none(self, vs_manager: VectorSearchManager) -> None:
        mock_index = _stub_index(vs_manager)
        vs_manager.search("query")
        assert "filters" not in mock_index.similarity_search.call_args.kwargs

    def test_hybrid_query_type_included_when_set(
        self, vs_manager: VectorSearchManager
    ) -> None:
        mock_index = _stub_index(vs_manager)
        vs_manager.search("query", query_type="hybrid")
        assert mock_index.similarity_search.call_args.kwargs["query_type"] == "hybrid"

    def test_query_type_omitted_when_none(self, vs_manager: VectorSearchManager) -> None:
        mock_index = _stub_index(vs_manager)
        vs_manager.search("query")
        assert "query_type" not in mock_index.similarity_search.call_args.kwargs

    def test_custom_columns_override_defaults(
        self, vs_manager: VectorSearchManager
    ) -> None:
        mock_index = _stub_index(vs_manager)
        custom_cols = ["id", "text"]
        vs_manager.search("query", columns=custom_cols)
        assert mock_index.similarity_search.call_args.kwargs["columns"] == custom_cols


# VectorSearchManager.create_endpoint_if_not_exists


class TestCreateEndpointIfNotExists:
    """Endpoint creation is skipped when the endpoint already exists."""

    @patch(_PATCH_CLIENT)
    def test_skips_creation_when_endpoint_exists(
        self, mock_client_cls: MagicMock, sample_config: ProjectConfig
    ) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.list_endpoints.return_value = {
            "endpoints": [{"name": sample_config.vector_search_endpoint}]
        }
        mgr = VectorSearchManager(config=sample_config)
        mgr.create_endpoint_if_not_exists()
        mock_client.create_endpoint_and_wait.assert_not_called()

    @patch(_PATCH_CLIENT)
    def test_creates_endpoint_when_absent(
        self, mock_client_cls: MagicMock, sample_config: ProjectConfig
    ) -> None:
        mock_client = mock_client_cls.return_value
        mock_client.list_endpoints.return_value = {"endpoints": []}
        mgr = VectorSearchManager(config=sample_config)
        mgr.create_endpoint_if_not_exists()
        mock_client.create_endpoint_and_wait.assert_called_once()
        create_kwargs = mock_client.create_endpoint_and_wait.call_args.kwargs
        assert create_kwargs["name"] == sample_config.vector_search_endpoint
        assert create_kwargs["endpoint_type"] == "STANDARD"

    @patch(_PATCH_CLIENT)
    def test_handles_missing_endpoints_key(
        self, mock_client_cls: MagicMock, sample_config: ProjectConfig
    ) -> None:
        """A response with no 'endpoints' key must not raise."""
        mock_client = mock_client_cls.return_value
        mock_client.list_endpoints.return_value = {}  # no 'endpoints' key
        mgr = VectorSearchManager(config=sample_config)
        mgr.create_endpoint_if_not_exists()  # should not raise
        mock_client.create_endpoint_and_wait.assert_called_once()
