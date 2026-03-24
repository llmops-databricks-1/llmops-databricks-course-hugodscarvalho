# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.3: Chunking Strategies
# MAGIC
# MAGIC ## Topics Covered
# MAGIC - Why chunking matters for RAG applications
# MAGIC - Different chunking approaches (fixed-size, sentence, paragraph, semantic, AI Parse)
# MAGIC - Extracting and cleaning chunks from AI-parsed EU legislation
# MAGIC - Chunk statistics and quality analysis
# MAGIC
# MAGIC ### Pipeline Context
# MAGIC
# MAGIC ```
# MAGIC ai_parsed_docs table (from 2.2)
# MAGIC     ↓  extract chunks + clean text + join metadata
# MAGIC eu_policy_chunks table (with Change Data Feed)
# MAGIC ```

# COMMAND ----------

import re

from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col

from eu_policy_agent.config import get_env, load_config
from eu_policy_agent.data_processor import DataProcessor

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)
catalog = cfg.catalog
schema = cfg.schema

logger.info(f"Catalog: {catalog}, Schema: {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Why Chunking Matters
# MAGIC
# MAGIC **Chunking** is the process of breaking documents into smaller pieces for:
# MAGIC
# MAGIC 1. **Embedding generation** — most embedding models have token limits (512–8 192 tokens)
# MAGIC 2. **Retrieval precision** — smaller chunks = more precise retrieval
# MAGIC 3. **Context-window management** — LLMs have limited context windows
# MAGIC 4. **Cost optimisation** — fewer tokens = lower inference cost
# MAGIC
# MAGIC ### The Chunking Trade-off
# MAGIC
# MAGIC | Direction | Benefit | Risk |
# MAGIC |-----------|---------|------|
# MAGIC | **Larger chunks** | More context per retrieval | Less precise, risk of "lost in the middle" |
# MAGIC | **Smaller chunks** | More precise retrieval | May lose surrounding context |
# MAGIC
# MAGIC **Optimal chunk size**: 256–512 tokens for most RAG use cases.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Chunking Strategies Overview
# MAGIC
# MAGIC ### Strategy 1: Fixed-Size Chunking
# MAGIC - Split by character count or token count
# MAGIC - Simple and fast
# MAGIC - May break sentences or paragraphs
# MAGIC
# MAGIC ### Strategy 2: Sentence-Based Chunking
# MAGIC - Split on sentence boundaries
# MAGIC - Preserves semantic units
# MAGIC - Variable chunk sizes
# MAGIC
# MAGIC ### Strategy 3: Paragraph-Based Chunking
# MAGIC - Split on paragraph boundaries (double newline)
# MAGIC - Larger semantic units
# MAGIC - Better for documents with clear structure
# MAGIC
# MAGIC ### Strategy 4: Semantic Chunking
# MAGIC - Use AI to identify topic boundaries
# MAGIC - Most intelligent but slowest
# MAGIC - Best for complex documents
# MAGIC
# MAGIC ### Strategy 5: AI Parse Documents (Databricks) ← **our approach**
# MAGIC - AI identifies document structure
# MAGIC - Extracts elements (text, tables, headings, …)
# MAGIC - Each element becomes a chunk
# MAGIC - Ideal for structured EU legislation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Process Chunks from Parsed PDFs
# MAGIC
# MAGIC The `DataProcessor.process_chunks()` method:
# MAGIC 1. Reads parsed JSON from `ai_parsed_docs`
# MAGIC 2. Extracts text elements as individual chunks
# MAGIC 3. Cleans each chunk (fixes hyphenation, collapses whitespace)
# MAGIC 4. Joins with document metadata from `raw_documents`
# MAGIC 5. Writes to `eu_policy_chunks` with Change Data Feed enabled
# MAGIC 6. Marks source documents as `processed = True`

# COMMAND ----------

processor = DataProcessor(spark=spark, config=cfg)

# Run chunk extraction and storage
processor.process_chunks()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Inspect the Chunks Table

# COMMAND ----------

chunks_df = spark.table(processor.chunks_table_fqn)

logger.info(f"Total chunks: {chunks_df.count()}")
chunks_df.printSchema()

# COMMAND ----------

# Show a few sample chunks
chunks_df.select("id", "document_id", "title", "document_type", "year").show(
    10, truncate=60
)

# COMMAND ----------

# Preview actual chunk text
sample_chunks = chunks_df.select("document_id", "chunk_id", "text").limit(5).collect()
for row in sample_chunks:
    preview = row["text"][:300] if row["text"] else "(empty)"
    logger.info(f"\n[{row['document_id']} / chunk {row['chunk_id']}]\n  {preview}…")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Chunk Statistics

# COMMAND ----------

chunk_stats = chunks_df.select(
    F.count("*").alias("total_chunks"),
    F.avg(F.length(col("text"))).alias("avg_length_chars"),
    F.min(F.length(col("text"))).alias("min_length_chars"),
    F.max(F.length(col("text"))).alias("max_length_chars"),
    F.stddev(F.length(col("text"))).alias("stddev_length_chars"),
).collect()[0]

logger.info("Chunk Statistics:")
logger.info(f"  Total chunks       : {chunk_stats['total_chunks']}")
logger.info(f"  Avg length (chars) : {chunk_stats['avg_length_chars']:.0f}")
logger.info(f"  Min length (chars) : {chunk_stats['min_length_chars']}")
logger.info(f"  Max length (chars) : {chunk_stats['max_length_chars']}")
logger.info(f"  Std-dev (chars)    : {chunk_stats['stddev_length_chars']:.0f}")

# Approximate token count (≈ 4 chars per token for English text)
avg_tokens = (chunk_stats["avg_length_chars"] or 0) / 4
logger.info(f"  Approx avg tokens  : {avg_tokens:.0f}")

# COMMAND ----------

# Chunks per document
logger.info("Chunks per document:")
(
    chunks_df.groupBy("document_id", "title")
    .count()
    .orderBy("count", ascending=False)
    .show(truncate=60)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Alternative Chunking Strategies (Illustrative)
# MAGIC
# MAGIC While we use AI Parse Documents for production, let's explore other strategies
# MAGIC applied to our EU policy text for comparison.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 1: Fixed-Size Chunking


# COMMAND ----------


def fixed_size_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Create fixed-size chunks with overlap.

    Args:
        text: Text to chunk.
        chunk_size: Size of each chunk in characters.
        overlap: Number of characters to overlap between chunks.

    Returns:
        List of text chunks.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# COMMAND ----------

# Demonstrate on a sample chunk
sample_text = chunks_df.select("text").first()
if sample_text and sample_text["text"]:
    fixed = fixed_size_chunking(sample_text["text"], chunk_size=500, overlap=50)
    logger.info(f"Original length: {len(sample_text['text'])} chars")
    logger.info(f"Fixed-size chunks: {len(fixed)}")
    logger.info(f"First chunk preview: {fixed[0][:200]}…")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Strategy 2: Sentence-Based Chunking

# COMMAND ----------


def sentence_chunking(text: str, max_sentences: int = 5) -> list[str]:
    """Create chunks based on sentence boundaries.

    Args:
        text: Text to chunk.
        max_sentences: Maximum sentences per chunk.

    Returns:
        List of text chunks.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []

    for sentence in sentences:
        current.append(sentence)
        if len(current) >= max_sentences:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

    return chunks


# COMMAND ----------

if sample_text and sample_text["text"]:
    sent_chunks = sentence_chunking(sample_text["text"], max_sentences=5)
    logger.info(f"Sentence-based chunks: {len(sent_chunks)}")
    logger.info(f"First chunk preview: {sent_chunks[0][:200]}…")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Chunk Size Recommendations
# MAGIC
# MAGIC | Use Case | Recommended Size | Reasoning |
# MAGIC |----------|-----------------|-----------|
# MAGIC | **Question Answering** | 256–512 tokens | Precise retrieval, focused answers |
# MAGIC | **Summarisation** | 512–1 024 tokens | More context needed |
# MAGIC | **Semantic Search** | 256–512 tokens | Balance precision vs. context |
# MAGIC | **Legal / Policy analysis** | 512–1 024 tokens | Preserve article context |
# MAGIC
# MAGIC **Token estimation**: ≈ 4 characters = 1 token (English text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Best Practices
# MAGIC
# MAGIC ### ✅ Do
# MAGIC 1. **Clean text** before chunking (remove extra whitespace, fix hyphenation)
# MAGIC 2. **Preserve metadata** (document_id, title, year, regulation_number, etc.)
# MAGIC 3. **Test different chunk sizes** for your specific use case
# MAGIC 4. **Use overlap** for better context (50–100 characters)
# MAGIC 5. **Monitor chunk quality** (length distribution, content quality)
# MAGIC
# MAGIC ### ❌ Don't
# MAGIC 1. Split in the middle of sentences (unless using fixed-size deliberately)
# MAGIC 2. Ignore document structure (articles, numbered paragraphs)
# MAGIC 3. Forget to clean and normalise text
# MAGIC 4. Lose metadata during chunking
# MAGIC 5. Use the same chunk size for all document types

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook we:
# MAGIC 1. ✅ Explored different chunking strategies
# MAGIC 2. ✅ Extracted and cleaned chunks from AI-parsed EU legislation
# MAGIC 3. ✅ Joined chunks with document metadata
# MAGIC 4. ✅ Stored chunks in `eu_policy_chunks` with Change Data Feed
# MAGIC 5. ✅ Analysed chunk statistics and distribution
# MAGIC
# MAGIC **Next**: Notebook 2.4 — Embeddings & Vector Search

# COMMAND ----------
