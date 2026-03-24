"""Data processing pipeline for EU policy documents.

Pipeline flow:
    raw_documents table (from 1.3 ingestion)
        ↓  (parse_pdfs_with_ai)
    ai_parsed_docs table (JSON from ai_parse_document)
        ↓  (process_chunks)
    eu_policy_chunks table (clean text + metadata)
        ↓  (VectorSearchManager - separate module)
    Vector Search Index (embeddings)
"""

import json
import re

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    concat_ws,
    current_timestamp,
    explode,
    udf,
)
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from eu_policy_agent.config import ProjectConfig


class DataProcessor:
    """Processes EU policy PDFs into searchable text chunks.

    Handles the complete workflow of:
    - Identifying unprocessed documents from the ``raw_documents`` table
    - Parsing PDFs with Databricks ``ai_parse_document``
    - Extracting and cleaning text chunks
    - Joining chunks with document metadata
    - Saving chunks to a Delta table with Change Data Feed enabled
    """

    # Table names (relative to catalog.schema)
    RAW_DOCUMENTS_TABLE: str = "raw_documents"
    PARSED_TABLE: str = "ai_parsed_docs"
    CHUNKS_TABLE: str = "eu_policy_chunks"

    def __init__(
        self,
        spark: SparkSession,
        config: ProjectConfig,
    ) -> None:
        """Initialise the processor.

        Args:
            spark: Active SparkSession (local or remote via Databricks Connect).
            config: Resolved ``ProjectConfig`` for the current environment.
        """
        self.spark = spark
        self.cfg = config
        self.catalog = config.catalog
        self.schema = config.schema

    # Fully-qualified table helpers

    @property
    def raw_documents_fqn(self) -> str:
        """Fully-qualified name of the raw documents table."""
        return f"{self.catalog}.{self.schema}.{self.RAW_DOCUMENTS_TABLE}"

    @property
    def parsed_table_fqn(self) -> str:
        """Fully-qualified name of the AI-parsed docs table."""
        return f"{self.catalog}.{self.schema}.{self.PARSED_TABLE}"

    @property
    def chunks_table_fqn(self) -> str:
        """Fully-qualified name of the chunks table."""
        return f"{self.catalog}.{self.schema}.{self.CHUNKS_TABLE}"

    # Step 1 - Parse PDFs using ai_parse_document

    def parse_pdfs_with_ai(self) -> int:
        """Parse unprocessed PDFs using ``ai_parse_document``.

        Reads PDFs from the volume paths stored in the ``raw_documents``
        table, parses each file with the Databricks AI document parser, and
        writes the raw JSON output to the ``ai_parsed_docs`` table.

        Returns:
            Number of documents parsed in this run.
        """
        # Ensure the parsed-docs table exists
        self.spark.sql(f"""
            CREATE TABLE IF NOT EXISTS {self.parsed_table_fqn} (
                document_id STRING,
                volume_path STRING,
                parsed_content STRING,
                parsed_at TIMESTAMP
            )
        """)

        # Identify documents that haven't been parsed yet
        unprocessed_df = (
            self.spark.table(self.raw_documents_fqn)
            .alias("raw")
            .join(
                self.spark.table(self.parsed_table_fqn).alias("parsed"),
                col("raw.document_id") == col("parsed.document_id"),
                "left_anti",
            )
            .select("raw.document_id", "raw.volume_path")
        )

        doc_count: int = unprocessed_df.count()
        if doc_count == 0:
            logger.info("All documents have already been parsed - nothing to do.")
            return 0

        logger.info(f"Parsing {doc_count} unprocessed document(s)…")

        # We iterate per document to keep track of which document_id maps
        # to which parsed output.  We read each PDF as a binary file,
        # register it as a temp view, and call ai_parse_document via SQL.
        rows = unprocessed_df.collect()
        for row in rows:
            doc_id: str = row["document_id"]
            vol_path: str = row["volume_path"]

            logger.info(f"  Parsing {doc_id} ({vol_path})…")

            # Read the single PDF as a binary file DataFrame
            binary_df = self.spark.read.format("binaryFile").load(vol_path)
            binary_df.createOrReplaceTempView("_tmp_binary_pdf")

            self.spark.sql(f"""
                INSERT INTO {self.parsed_table_fqn}
                SELECT
                    '{doc_id}'              AS document_id,
                    path                    AS volume_path,
                    ai_parse_document(content) AS parsed_content,
                    current_timestamp()     AS parsed_at
                FROM _tmp_binary_pdf
            """)

        logger.info(f"✓ Parsed {doc_count} document(s) into {self.parsed_table_fqn}")
        return doc_count

    # Step 2 - Extract, clean, and store chunks

    @staticmethod
    def _extract_chunks(
        parsed_content_json: str,
    ) -> list[tuple[str, str]]:
        """Extract text chunks from the parsed-document JSON.

        The ``ai_parse_document`` output contains a ``document.elements``
        list.  Each element with ``type == "text"`` becomes a chunk.

        Args:
            parsed_content_json: Raw JSON string from ai_parse_document.

        Returns:
            List of ``(chunk_id, content)`` tuples.
        """
        parsed = json.loads(parsed_content_json)
        elements = parsed.get("document", {}).get("elements", [])
        return [
            (elem.get("id", ""), elem.get("content", ""))
            for elem in elements
            if elem.get("type") == "text"
        ]

    @staticmethod
    def _clean_chunk(text: str) -> str:
        """Clean and normalise a raw text chunk.

        Applies the following transformations:
        1. Fix hyphenation across line breaks (``docu-\\nments`` → ``documents``)
        2. Collapse internal newlines into single spaces
        3. Collapse repeated whitespace

        Args:
            text: Raw text content.

        Returns:
            Cleaned text string.
        """
        # Fix hyphenation across line breaks
        t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
        # Collapse internal newlines into spaces
        t = re.sub(r"\s*\n\s*", " ", t)
        # Collapse repeated whitespace
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    def process_chunks(self) -> None:
        """Extract, clean, and persist chunks from parsed documents.

        Reads from ``ai_parsed_docs``, extracts text elements, cleans them,
        joins with document metadata from ``raw_documents``, and writes to
        the ``eu_policy_chunks`` Delta table with Change Data Feed enabled.
        """
        parsed_df = self.spark.table(self.parsed_table_fqn)

        if parsed_df.count() == 0:
            logger.info("No parsed documents to process.")
            return

        # UDF registration
        chunk_schema = ArrayType(
            StructType(
                [
                    StructField("chunk_id", StringType(), nullable=True),
                    StructField("content", StringType(), nullable=True),
                ]
            )
        )
        extract_chunks_udf = udf(self._extract_chunks, chunk_schema)
        clean_chunk_udf = udf(self._clean_chunk, StringType())

        # Metadata from raw_documents
        metadata_df = self.spark.table(self.raw_documents_fqn).select(
            col("document_id"),
            col("title"),
            col("official_title"),
            col("document_type"),
            col("regulation_number"),
            col("year"),
            concat_ws(", ", col("topics")).alias("topics"),
            col("official_url"),
        )

        # Chunk extraction pipeline
        chunks_df = (
            parsed_df.withColumn("chunks", extract_chunks_udf(col("parsed_content")))
            .withColumn("chunk", explode(col("chunks")))
            .select(
                col("document_id"),
                col("chunk.chunk_id").alias("chunk_id"),
                clean_chunk_udf(col("chunk.content")).alias("text"),
                # Unique surrogate key: <document_id>_<chunk_id>
                concat_ws("_", col("document_id"), col("chunk.chunk_id")).alias("id"),
            )
            .join(metadata_df, on="document_id", how="left")
            .withColumn("processed_at", current_timestamp())
        )

        # Write to Delta
        chunks_df.write.mode("append").saveAsTable(self.chunks_table_fqn)
        logger.info(f"✓ Saved chunks to {self.chunks_table_fqn}")

        # Enable Change Data Feed (idempotent)
        self.spark.sql(f"""
            ALTER TABLE {self.chunks_table_fqn}
            SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
        """)
        logger.info(f"✓ Change Data Feed enabled for {self.chunks_table_fqn}")

        # Mark source documents as processed
        processed_ids = parsed_df.select("document_id").distinct().collect()
        id_list = ", ".join(f"'{r['document_id']}'" for r in processed_ids)
        self.spark.sql(f"""
            UPDATE {self.raw_documents_fqn}
            SET processed = true
            WHERE document_id IN ({id_list})
        """)
        logger.info(
            f"✓ Marked {len(processed_ids)} document(s) as processed "
            f"in {self.raw_documents_fqn}"
        )

    # Convenience - run the full pipeline

    def process_and_save(self) -> None:
        """Run the complete data processing pipeline.

        1. Parse unprocessed PDFs with ``ai_parse_document``
        2. Extract, clean, and store chunks with metadata
        """
        parsed_count = self.parse_pdfs_with_ai()
        if parsed_count == 0:
            logger.info("No new documents to process - pipeline complete.")
            return

        self.process_chunks()
        logger.info("✓ Data processing pipeline complete!")
