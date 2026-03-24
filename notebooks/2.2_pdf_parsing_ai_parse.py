# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.2: PDF Parsing with AI Parse Documents
# MAGIC
# MAGIC ## Topics Covered
# MAGIC - AI Parse Documents for intelligent PDF parsing
# MAGIC - Comparison with other PDF parsing tools
# MAGIC - Parsing EU legislation PDFs and storing structured output
# MAGIC
# MAGIC ### Pipeline Context
# MAGIC
# MAGIC ```
# MAGIC raw_documents table (from 1.3)
# MAGIC     ↓  ai_parse_document()
# MAGIC ai_parsed_docs table (JSON)
# MAGIC     ↓  (next notebook: 2.3)
# MAGIC eu_policy_chunks table
# MAGIC ```

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession

from eu_policy_agent.config import get_env, load_config
from eu_policy_agent.data_processor import DataProcessor

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)

logger.info(f"Environment : {env}")
logger.info(f"Catalog     : {cfg.catalog}")
logger.info(f"Schema      : {cfg.schema}")
logger.info(f"Volume      : {cfg.volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. PDF Parsing Tools Comparison
# MAGIC
# MAGIC | Tool | Pros | Cons | Best For |
# MAGIC |------|------|------|----------|
# MAGIC | **AI Parse Documents** | AI-powered · handles complex layouts · Databricks-native · preserves structure | Databricks-specific · cost per page | Complex documents, tables, multi-column |
# MAGIC | **PyPDF2 / pypdf** | Simple · free · pure Python | Poor with complex layouts · no table extraction | Simple text extraction |
# MAGIC | **pdfplumber** | Good table extraction · layout analysis | Slower · manual tuning needed | Tables and structured data |
# MAGIC | **Apache Tika** | Multi-format support · metadata extraction | Java dependency · heavy | Multi-format processing |
# MAGIC | **Unstructured.io** | ML-powered · good chunking | External service · API costs | Modern RAG pipelines |
# MAGIC
# MAGIC **AI Parse Documents** is the recommended choice for Databricks users due to its
# MAGIC integration with Unity Catalog and intelligent layout handling — which matters
# MAGIC for EU legislation that contains complex articles, numbered paragraphs, and tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Review Source Data
# MAGIC
# MAGIC The `raw_documents` table was populated in **Notebook 1.3**.
# MAGIC Let's verify what we have before parsing.

# COMMAND ----------

processor = DataProcessor(spark=spark, config=cfg)

# Show documents available for parsing
raw_df = spark.table(processor.raw_documents_fqn)

logger.info(f"Total documents in {processor.raw_documents_fqn}: {raw_df.count()}")
raw_df.select(
    "document_id",
    "document_type",
    "regulation_number",
    "year",
    "num_pages",
    "processed",
).show(truncate=60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Parse PDFs with AI Parse Documents
# MAGIC
# MAGIC The `DataProcessor.parse_pdfs_with_ai()` method:
# MAGIC 1. Identifies unprocessed documents (those not yet in `ai_parsed_docs`)
# MAGIC 2. Reads each PDF binary from the Unity Catalog Volume
# MAGIC 3. Calls `ai_parse_document(content, 'TEXT')` to extract structured JSON
# MAGIC 4. Stores the result in the `ai_parsed_docs` Delta table
# MAGIC
# MAGIC The SQL function `ai_parse_document` is a Databricks-native AI function that
# MAGIC uses a layout-aware model to identify text blocks, tables, and headings.

# COMMAND ----------

parsed_count = processor.parse_pdfs_with_ai()
logger.info(f"Documents parsed in this run: {parsed_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Inspect Parsed Output
# MAGIC
# MAGIC Each row in `ai_parsed_docs` contains the full JSON output from
# MAGIC `ai_parse_document`.  Let's inspect the structure.

# COMMAND ----------

parsed_df = spark.table(processor.parsed_table_fqn)
logger.info(f"Total parsed documents: {parsed_df.count()}")

parsed_df.select("document_id", "volume_path", "parsed_at").show(truncate=80)

# COMMAND ----------

# Preview the JSON structure for one document
import json

sample_row = parsed_df.select("document_id", "parsed_content").first()
if sample_row:
    doc_id = sample_row["document_id"]
    parsed_json = json.loads(sample_row["parsed_content"])

    # Show high-level keys
    logger.info(f"Document: {doc_id}")
    logger.info(f"Top-level keys: {list(parsed_json.keys())}")

    elements = parsed_json.get("document", {}).get("elements", [])
    logger.info(f"Total elements: {len(elements)}")

    # Count by element type
    from collections import Counter

    type_counts = Counter(e.get("type", "unknown") for e in elements)
    logger.info(f"Element types: {dict(type_counts)}")

    # Show first 3 text elements as a preview
    text_elements = [e for e in elements if e.get("type") == "text"][:3]
    for i, elem in enumerate(text_elements, 1):
        preview = elem.get("content", "")[:200]
        logger.info(f"\n  Text element {i} (id={elem.get('id', '?')}):")
        logger.info(f"    {preview}…")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Next Steps
# MAGIC
# MAGIC The parsed JSON is now stored in the `ai_parsed_docs` table.
# MAGIC In **Notebook 2.3** we will:
# MAGIC - Extract text chunks from the JSON structure
# MAGIC - Clean and normalise the text
# MAGIC - Join chunks with document metadata
# MAGIC - Store the result in the `eu_policy_chunks` table

# COMMAND ----------
