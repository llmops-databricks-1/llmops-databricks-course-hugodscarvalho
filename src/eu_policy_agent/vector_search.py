"""Vector search management for EU policy document chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from databricks.vector_search.client import VectorSearchClient
from loguru import logger

if TYPE_CHECKING:
    from eu_policy_agent.config import ProjectConfig


class VectorSearchManager:
    """Create and manage a Databricks Vector Search index over EU policy chunks.

    Responsibilities:
    - Provision (or reuse) a Vector Search endpoint
    - Create (or reuse) a Delta Sync index backed by the chunks table
    - Trigger index syncs after new data is written
    - Execute similarity, hybrid, and reranked searches
    """

    # Default source table and index names (relative to catalog.schema)
    _SOURCE_TABLE: str = "eu_policy_chunks"
    _INDEX_SUFFIX: str = "eu_policy_index"

    def __init__(
        self,
        config: ProjectConfig,
        endpoint_name: str | None = None,
        embedding_model: str | None = None,
        usage_policy_id: str | None = None,
    ) -> None:
        """Initialise the manager.

        Args:
            config: Resolved ``ProjectConfig`` for the active environment.
            endpoint_name: Override the Vector Search endpoint name
                (defaults to ``config.vector_search_endpoint``).
            embedding_model: Override the embedding model endpoint name
                (defaults to ``config.embedding_endpoint``).
            usage_policy_id: Optional Databricks usage-policy ID for
                the endpoint.
        """
        self.config = config
        self.endpoint_name = endpoint_name or config.vector_search_endpoint
        self.embedding_model = embedding_model or config.embedding_endpoint
        self.catalog = config.catalog
        self.schema = config.schema
        self.usage_policy_id = usage_policy_id

        self.client = VectorSearchClient()
        self.source_table = f"{self.catalog}.{self.schema}.{self._SOURCE_TABLE}"
        self.index_name = f"{self.catalog}.{self.schema}.{self._INDEX_SUFFIX}"

    # Endpoint management

    def create_endpoint_if_not_exists(self) -> None:
        """Create the Vector Search endpoint if it does not already exist.

        Uses the ``STANDARD`` endpoint type which is suitable for
        development and most production workloads.
        """
        endpoints_response = self.client.list_endpoints()
        endpoints = (
            endpoints_response.get("endpoints", [])
            if isinstance(endpoints_response, dict)
            else []
        )
        endpoint_exists = any(
            (ep.get("name") if isinstance(ep, dict) else getattr(ep, "name", None))
            == self.endpoint_name
            for ep in endpoints
        )

        if endpoint_exists:
            logger.info(f"✓ Vector Search endpoint already exists: {self.endpoint_name}")
            return

        logger.info(f"Creating Vector Search endpoint: {self.endpoint_name}…")
        self.client.create_endpoint_and_wait(
            name=self.endpoint_name,
            endpoint_type="STANDARD",
            usage_policy_id=self.usage_policy_id,
        )
        logger.info(f"✓ Vector Search endpoint created: {self.endpoint_name}")

    # Index management

    def create_or_get_index(self) -> object:
        """Create (or retrieve) the Delta Sync vector-search index.

        The index is configured with:
        - ``pipeline_type="TRIGGERED"`` - sync on demand, ideal for
          batch pipelines.
        - ``primary_key="id"`` - the surrogate key
          ``<document_id>_<chunk_id>``.
        - ``embedding_source_column="text"`` - the cleaned chunk text.
        - ``embedding_model_endpoint_name`` - the model from config.

        Returns:
            A ``VectorSearchIndex`` handle.
        """
        self.create_endpoint_if_not_exists()

        # Try to retrieve an existing index first
        try:
            index = self.client.get_index(index_name=self.index_name)
            logger.info(f"✓ Vector Search index already exists: {self.index_name}")
            return index
        except Exception:
            logger.info(f"Index {self.index_name} not found - creating…")

        # Create a new Delta Sync index
        try:
            index = self.client.create_delta_sync_index(
                endpoint_name=self.endpoint_name,
                source_table_name=self.source_table,
                index_name=self.index_name,
                pipeline_type="TRIGGERED",
                primary_key="id",
                embedding_source_column="text",
                embedding_model_endpoint_name=self.embedding_model,
                usage_policy_id=self.usage_policy_id,
            )
            logger.info(f"✓ Vector Search index created: {self.index_name}")
            return index
        except Exception as e:
            if "RESOURCE_ALREADY_EXISTS" not in str(e):
                raise
            # Race condition - index was created between our check and create
            logger.info(f"✓ Vector Search index already exists: {self.index_name}")
            return self.client.get_index(index_name=self.index_name)

    # Sync

    def sync_index(self, *, max_retries: int = 6, backoff_seconds: int = 10) -> None:
        """Trigger an index sync against the source Delta table.

        If the endpoint is still provisioning, retries with exponential
        backoff up to ``max_retries`` times.

        Args:
            max_retries: Maximum number of retry attempts.
            backoff_seconds: Initial wait between retries (doubles each time).
        """
        import time

        index = self.create_or_get_index()

        wait = backoff_seconds
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Syncing index {self.index_name}…")
                index.sync()
                logger.info("✓ Index sync triggered")
                return
            except Exception as exc:
                if "not ready yet" in str(exc) and attempt < max_retries:
                    logger.warning(
                        f"Endpoint not ready - retrying in {wait}s "
                        f"(attempt {attempt}/{max_retries})"
                    )
                    time.sleep(wait)
                    wait *= 2
                else:
                    raise

    # Search helpers

    @staticmethod
    def parse_results(results: dict) -> list[dict]:
        """Convert raw Vector Search results into a list of dictionaries.

        Args:
            results: Raw output from ``index.similarity_search()``.

        Returns:
            List of row dictionaries with column names as keys.
        """
        columns = [c["name"] for c in results.get("manifest", {}).get("columns", [])]
        data_array = results.get("result", {}).get("data_array", [])
        return [dict(zip(columns, row, strict=True)) for row in data_array]

    def search(
        self,
        query: str,
        *,
        num_results: int = 5,
        columns: list[str] | None = None,
        filters: dict | None = None,
        query_type: str | None = None,
        reranker: object | None = None,
    ) -> dict:
        """Execute a similarity search against the index.

        Args:
            query: Natural-language search query.
            num_results: Maximum number of results to return.
            columns: Columns to include in the response. Defaults to a
                sensible set of chunk + metadata fields.
            filters: Optional metadata filters (e.g. ``{"year": "2024"}``).
            query_type: ``None`` for pure semantic, ``"hybrid"`` for
                semantic + keyword (BM25).
            reranker: Optional ``DatabricksReranker`` instance for
                two-stage retrieval.

        Returns:
            Raw results dictionary from Vector Search.
        """
        if columns is None:
            columns = [
                "id",
                "text",
                "document_id",
                "title",
                "document_type",
                "year",
            ]

        index = self.client.get_index(index_name=self.index_name)

        kwargs: dict[str, Any] = {
            "query_text": query,
            "columns": columns,
            "num_results": num_results,
        }
        if filters is not None:
            kwargs["filters"] = filters
        if query_type is not None:
            kwargs["query_type"] = query_type
        if reranker is not None:
            kwargs["reranker"] = reranker

        return index.similarity_search(**kwargs)
