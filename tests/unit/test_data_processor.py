"""Unit tests for eu_policy_agent.data_processor.

Strategy
--------
DataProcessor relies on a live SparkSession for its pipeline methods.
We test the two static utility methods - _extract_chunks and _clean_chunk -
in isolation (no Spark required), and verify the fully-qualified table name
properties using a mock SparkSession so we never hit a real cluster.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from eu_policy_agent.config import ProjectConfig
from eu_policy_agent.data_processor import DataProcessor

# Helpers


def _make_parsed_json(elements: list[dict]) -> str:  # noqa: ANN401
    """Wrap a list of elements in the ai_parse_document JSON envelope."""
    return json.dumps({"document": {"elements": elements}})


# DataProcessor._extract_chunks


class TestExtractChunks:
    """_extract_chunks parses the ai_parse_document JSON into (id, content) tuples."""

    def test_returns_text_elements(self) -> None:
        payload = _make_parsed_json(
            [
                {"id": "el-1", "type": "text", "content": "Hello world"},
                {"id": "el-2", "type": "text", "content": "Second paragraph"},
            ]
        )
        result = DataProcessor._extract_chunks(payload)
        assert result == [("el-1", "Hello world"), ("el-2", "Second paragraph")]

    def test_filters_out_non_text_elements(self) -> None:
        payload = _make_parsed_json(
            [
                {"id": "t-1", "type": "text", "content": "Keep me"},
                {"id": "i-1", "type": "image", "content": "drop_this"},
                {"id": "t-2", "type": "text", "content": "Keep me too"},
            ]
        )
        result = DataProcessor._extract_chunks(payload)
        assert len(result) == 2
        assert all(cid in ("t-1", "t-2") for cid, _ in result)

    def test_empty_elements_list(self) -> None:
        payload = _make_parsed_json([])
        assert DataProcessor._extract_chunks(payload) == []

    def test_empty_document_key(self) -> None:
        payload = json.dumps({"document": {}})
        assert DataProcessor._extract_chunks(payload) == []

    def test_missing_document_key(self) -> None:
        payload = json.dumps({})
        assert DataProcessor._extract_chunks(payload) == []

    def test_element_with_missing_id_defaults_to_empty_string(self) -> None:
        payload = _make_parsed_json([{"type": "text", "content": "No id here"}])
        result = DataProcessor._extract_chunks(payload)
        assert result == [("", "No id here")]

    def test_element_with_missing_content_defaults_to_empty_string(self) -> None:
        payload = _make_parsed_json([{"id": "el-1", "type": "text"}])
        result = DataProcessor._extract_chunks(payload)
        assert result == [("el-1", "")]

    def test_no_text_elements_returns_empty_list(self) -> None:
        payload = _make_parsed_json([{"id": "h-1", "type": "header", "content": "Title"}])
        assert DataProcessor._extract_chunks(payload) == []


# DataProcessor._clean_chunk


class TestCleanChunk:
    """_clean_chunk normalises raw text from parsed PDF elements."""

    def test_fixes_hyphenation_across_line_break(self) -> None:
        raw = "docu-\nments"
        assert DataProcessor._clean_chunk(raw) == "documents"

    def test_collapses_newlines_into_spaces(self) -> None:
        raw = "First line\nSecond line"
        assert DataProcessor._clean_chunk(raw) == "First line Second line"

    def test_collapses_repeated_whitespace(self) -> None:
        raw = "too   many   spaces"
        assert DataProcessor._clean_chunk(raw) == "too many spaces"

    def test_strips_leading_and_trailing_whitespace(self) -> None:
        raw = "   leading and trailing   "
        assert DataProcessor._clean_chunk(raw) == "leading and trailing"

    def test_combined_transformations(self) -> None:
        raw = "  pro-\ncess   multiple\n\n  spaces  "
        assert DataProcessor._clean_chunk(raw) == "process multiple spaces"

    def test_clean_text_is_unchanged(self) -> None:
        raw = "This is already clean text."
        assert DataProcessor._clean_chunk(raw) == raw

    def test_empty_string_returns_empty_string(self) -> None:
        assert DataProcessor._clean_chunk("") == ""


# DataProcessor - fully-qualified table name properties


class TestDataProcessorFQNProperties:
    """FQN properties build correct three-part table names."""

    @pytest.fixture()
    def processor(self, sample_config: ProjectConfig) -> DataProcessor:
        return DataProcessor(spark=MagicMock(), config=sample_config)

    def test_raw_documents_fqn(self, processor: DataProcessor) -> None:
        assert processor.raw_documents_fqn == ("test_catalog.test_schema.raw_documents")

    def test_parsed_table_fqn(self, processor: DataProcessor) -> None:
        assert processor.parsed_table_fqn == ("test_catalog.test_schema.ai_parsed_docs")

    def test_chunks_table_fqn(self, processor: DataProcessor) -> None:
        assert processor.chunks_table_fqn == ("test_catalog.test_schema.eu_policy_chunks")

    def test_fqn_reflects_config_catalog_and_schema(self) -> None:
        cfg = ProjectConfig(catalog="prod", schema="eu", volume="v")
        proc = DataProcessor(spark=MagicMock(), config=cfg)
        assert proc.raw_documents_fqn.startswith("prod.eu.")


# DataProcessor - process_and_save short-circuit (no new documents)


class TestProcessAndSaveShortCircuit:
    """process_and_save exits early when parse_pdfs_with_ai returns 0."""

    def test_process_chunks_not_called_when_nothing_parsed(
        self,
        sample_config: ProjectConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        proc = DataProcessor(spark=MagicMock(), config=sample_config)

        # Simulate: no new documents to parse
        monkeypatch.setattr(proc, "parse_pdfs_with_ai", lambda: 0)

        chunks_called: list[bool] = []
        monkeypatch.setattr(proc, "process_chunks", lambda: chunks_called.append(True))

        proc.process_and_save()

        assert chunks_called == [], (
            "process_chunks should not be called when parse_pdfs_with_ai returns 0"
        )
