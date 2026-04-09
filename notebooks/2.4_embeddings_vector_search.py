# Databricks notebook source
# MAGIC %md
# MAGIC # Lecture 2.4: Embeddings & Vector Search
# MAGIC
# MAGIC ## Topics Covered
# MAGIC - Understanding embeddings and vector representations
# MAGIC - Embedding model comparison
# MAGIC - Creating a Vector Search endpoint and index
# MAGIC - Similarity search, hybrid search, and reranking
# MAGIC - Metadata filtering and search-quality comparison
# MAGIC
# MAGIC ### Pipeline Context
# MAGIC
# MAGIC ```
# MAGIC eu_policy_chunks table (from 2.3)
# MAGIC     ↓  Delta Sync + embedding model
# MAGIC Vector Search Index
# MAGIC     ↓  query
# MAGIC Search Results (scores + metadata)
# MAGIC ```

# COMMAND ----------

from databricks.vector_search.reranker import DatabricksReranker
from loguru import logger
from pyspark.sql import SparkSession

from eu_policy_agent.config import get_env, load_config
from eu_policy_agent.vector_search import VectorSearchManager

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../project_config.yml", env)
catalog = cfg.catalog
schema = cfg.schema

logger.info(f"Catalog: {catalog}, Schema: {schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Understanding Embeddings
# MAGIC
# MAGIC **Embeddings** are numerical representations of text that capture semantic meaning.
# MAGIC
# MAGIC ### Key Concepts
# MAGIC
# MAGIC - **Vector**: Array of numbers, e.g. `[0.1, -0.3, 0.5, …]`
# MAGIC - **Dimension**: Length of the vector (e.g. 384, 768, 1 024)
# MAGIC - **Semantic similarity**: Similar meanings → similar vectors
# MAGIC - **Distance metric**: Cosine similarity, Euclidean distance, dot product
# MAGIC
# MAGIC ### How It Works
# MAGIC
# MAGIC ```
# MAGIC Text: "data protection regulation"
# MAGIC   ↓ (Embedding Model)
# MAGIC Vector: [0.23, -0.15, 0.67, …, 0.42]   # 1 024 dimensions
# MAGIC
# MAGIC Text: "privacy law in Europe"
# MAGIC   ↓ (Embedding Model)
# MAGIC Vector: [0.25, -0.13, 0.65, …, 0.40]   # Very similar!
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Embedding Models Comparison
# MAGIC
# MAGIC | Model | Dimensions | Max Tokens | Best For |
# MAGIC |-------|-----------|------------|----------|
# MAGIC | **databricks-gte-large-en** | 1 024 | 512 | General purpose, fast, free on Databricks |
# MAGIC | **databricks-bge-large-en** | 1 024 | 512 | General purpose, high quality |
# MAGIC | **text-embedding-ada-002** (OpenAI) | 1 536 | 8 191 | High quality, higher cost |
# MAGIC | **e5-large-v2** | 1 024 | 512 | Open source, good quality |
# MAGIC | **all-MiniLM-L6-v2** | 384 | 512 | Fast, lightweight |
# MAGIC
# MAGIC We use **`databricks-gte-large-en`** — fast, high-quality, and included at
# MAGIC no extra cost on Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Vector Search Architecture
# MAGIC
# MAGIC ```
# MAGIC ┌──────────────────────────────────────────────┐
# MAGIC │     Delta Table (eu_policy_chunks)            │
# MAGIC │  - id               (primary key)             │
# MAGIC │  - text             (embedding source)        │
# MAGIC │  - document_id, title, year, …  (metadata)   │
# MAGIC └──────────────────┬───────────────────────────┘
# MAGIC                    │  Automatic delta sync
# MAGIC                    ↓
# MAGIC ┌──────────────────────────────────────────────┐
# MAGIC │     Vector Search Index                       │
# MAGIC │  - Embeddings generated automatically         │
# MAGIC │  - Stored in optimised ANN format             │
# MAGIC │  - Supports similarity + hybrid search        │
# MAGIC └──────────────────┬───────────────────────────┘
# MAGIC                    │  Query
# MAGIC                    ↓
# MAGIC ┌──────────────────────────────────────────────┐
# MAGIC │     Search Results                            │
# MAGIC │  - Most similar chunks with scores            │
# MAGIC │  - Metadata for filtering / display           │
# MAGIC └──────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Vector Search Endpoint & Index

# COMMAND ----------

vs_manager = VectorSearchManager(
    config=cfg,
    endpoint_name=cfg.vector_search_endpoint,
    embedding_model=cfg.embedding_endpoint,
)

logger.info(f"Endpoint       : {vs_manager.endpoint_name}")
logger.info(f"Embedding model: {vs_manager.embedding_model}")
logger.info(f"Index name     : {vs_manager.index_name}")
logger.info(f"Source table   : {vs_manager.source_table}")

# COMMAND ----------

# Create endpoint (if it doesn't already exist)
vs_manager.create_endpoint_if_not_exists()

# COMMAND ----------

# Create (or retrieve) the Delta Sync index
index = vs_manager.create_or_get_index()

logger.info("✓ Vector Search setup complete!")
logger.info(f"  Index : {vs_manager.index_name}")
logger.info(f"  Source: {vs_manager.source_table}")
logger.info(f"  Model : {vs_manager.embedding_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Index Configuration Notes
# MAGIC
# MAGIC | Option | Value | Explanation |
# MAGIC |--------|-------|-------------|
# MAGIC | `pipeline_type` | `TRIGGERED` | Sync on demand — ideal for batch pipelines |
# MAGIC | `primary_key` | `id` | `<document_id>_<chunk_id>` surrogate key |
# MAGIC | `embedding_source_column` | `text` | The cleaned chunk text |
# MAGIC | `embedding_model_endpoint_name` | `databricks-gte-large-en` | Free Databricks model |

# COMMAND ----------

# Trigger an initial sync so embeddings are computed
vs_manager.sync_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Semantic (Similarity) Search
# MAGIC
# MAGIC ### How It Works
# MAGIC
# MAGIC 1. **Query embedding** — convert the search text to a vector
# MAGIC 2. **Cosine similarity** — compare against all document vectors
# MAGIC 3. **Ranking** — return the top-k most similar chunks
# MAGIC
# MAGIC | Score Range | Interpretation |
# MAGIC |-------------|----------------|
# MAGIC | 0.8–1.0 | Very similar (near exact) |
# MAGIC | 0.5–0.7 | Somewhat related |
# MAGIC | < 0.5 | Less relevant |

# COMMAND ----------

query = "What are the obligations for high-risk AI systems?"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "document_id", "year"],
    num_results=5,
)

logger.info(f"Query: {query}\n")
logger.info("Top 5 Results:")
logger.info("=" * 80)

for i, row in enumerate(VectorSearchManager.parse_results(results), 1):
    logger.info(f"\n{i}. Document: {row.get('title', 'N/A')}")
    logger.info(f"   Doc ID  : {row.get('document_id', 'N/A')}")
    logger.info(f"   Year    : {row.get('year', 'N/A')}")
    logger.info(f"   Chunk ID: {row.get('id', 'N/A')}")
    logger.info(f"   Text    : {row.get('text', '')[:200]}…")
    logger.info(f"   Score   : {row.get('score', 'N/A'):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Metadata Filtering
# MAGIC
# MAGIC Narrow results to specific regulations, document types, or years.
# MAGIC
# MAGIC ```python
# MAGIC # Single filter
# MAGIC filters = {"year": "2024"}
# MAGIC
# MAGIC # Multiple filters (AND)
# MAGIC filters = {"document_type": "Regulation", "year": "2024"}
# MAGIC ```

# COMMAND ----------

query = "rules on personal data processing"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "year", "document_type"],
    filters={"document_type": "Regulation"},
    num_results=3,
)

logger.info(f"Query: {query}")
logger.info("Filter: document_type = 'Regulation'\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(VectorSearchManager.parse_results(results), 1):
    logger.info(f"\n{i}. {row.get('title', 'N/A')}")
    logger.info(f"   Year: {row.get('year', 'N/A')}")
    logger.info(f"   Type: {row.get('document_type', 'N/A')}")
    logger.info(f"   Text: {row.get('text', '')[:150]}…")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Hybrid Search: Semantic + Keyword (BM25)
# MAGIC
# MAGIC **Semantic search alone** may miss:
# MAGIC - Exact regulation numbers (e.g. "2024/1689")
# MAGIC - Specific legal terms or abbreviations
# MAGIC
# MAGIC **Hybrid search** combines:
# MAGIC - **Semantic search** (embeddings) → captures meaning, synonyms
# MAGIC - **Keyword search** (BM25) → exact term matching, TF-IDF scoring
# MAGIC
# MAGIC Merged via **Reciprocal Rank Fusion (RRF)**.

# COMMAND ----------

query = "GDPR data subject rights erasure"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title"],
    num_results=5,
    query_type="hybrid",
)

logger.info(f"Query: {query}")
logger.info("Search type: Hybrid (Semantic + Keyword)\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(VectorSearchManager.parse_results(results), 1):
    logger.info(f"\n{i}. {row.get('title', 'N/A')}")
    logger.info(f"   Text: {row.get('text', '')[:200]}…")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Reranking for Higher Precision
# MAGIC
# MAGIC ### Two-Stage Retrieval Pattern
# MAGIC
# MAGIC | Stage | Model | Purpose |
# MAGIC |-------|-------|---------|
# MAGIC | **1. Fast retrieval** (bi-encoder) | Embedding model | Get top 20–50 candidates quickly |
# MAGIC | **2. Precise reranking** (cross-encoder) | Reranker model | Score each candidate against the query |
# MAGIC
# MAGIC ### When to Use Reranking
# MAGIC
# MAGIC - High-stakes queries (legal, compliance)
# MAGIC - Complex multi-faceted questions
# MAGIC - When precision matters more than speed
# MAGIC
# MAGIC **Trade-off**: 10–30 % better relevance at ~2–5× latency.

# COMMAND ----------

query = "cybersecurity incident reporting obligations for essential entities"

results = index.similarity_search(
    query_text=query,
    columns=["text", "id", "title", "document_id", "official_title"],
    num_results=5,
    query_type="hybrid",
    reranker=DatabricksReranker(
        columns_to_rerank=["text", "title"],
    ),
)

logger.info(f"Query: {query}")
logger.info("Search type: Hybrid + Reranking\n")
logger.info("Results:")
logger.info("=" * 80)

for i, row in enumerate(VectorSearchManager.parse_results(results), 1):
    logger.info(f"\n{i}. {row.get('title', 'N/A')}")
    logger.info(f"   Text: {row.get('text', '')[:200]}…")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Search Quality Comparison
# MAGIC
# MAGIC Compare the three search strategies on the same query.

# COMMAND ----------

query = "gatekeeper obligations in digital markets"

logger.info(f"Query: {query}\n")

# Strategy 1: Basic semantic search
results_basic = index.similarity_search(
    query_text=query,
    columns=["text", "title"],
    num_results=3,
)

logger.info("Strategy 1: Basic Semantic Search")
logger.info("-" * 80)
for i, row in enumerate(VectorSearchManager.parse_results(results_basic), 1):
    logger.info(f"  {i}. {row.get('title', 'N/A')[:70]}…")

# Strategy 2: Hybrid search
results_hybrid = index.similarity_search(
    query_text=query,
    columns=["text", "title"],
    num_results=3,
    query_type="hybrid",
)

logger.info("\nStrategy 2: Hybrid Search")
logger.info("-" * 80)
for i, row in enumerate(VectorSearchManager.parse_results(results_hybrid), 1):
    logger.info(f"  {i}. {row.get('title', 'N/A')[:70]}…")

# Strategy 3: Hybrid + Reranking
results_reranked = index.similarity_search(
    query_text=query,
    columns=["text", "title"],
    num_results=3,
    query_type="hybrid",
    reranker=DatabricksReranker(columns_to_rerank=["text", "title"]),
)

logger.info("\nStrategy 3: Hybrid + Reranking")
logger.info("-" * 80)
for i, row in enumerate(VectorSearchManager.parse_results(results_reranked), 1):
    logger.info(f"  {i}. {row.get('title', 'N/A')[:70]}…")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Index Monitoring

# COMMAND ----------

index_info = vs_manager.client.get_index(
    endpoint_name=vs_manager.endpoint_name,
    index_name=vs_manager.index_name,
)

logger.info("Index Information:")
logger.info(f"  Name     : {index_info.name}")
logger.info(f"  Endpoint : {index_info.endpoint_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Index Maintenance
# MAGIC
# MAGIC ```python
# MAGIC # Trigger a manual sync (for TRIGGERED pipeline)
# MAGIC index.sync()
# MAGIC
# MAGIC # Delete index (if needed)
# MAGIC # vs_manager.client.delete_index(index_name=vs_manager.index_name)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Best Practices
# MAGIC
# MAGIC ### ✅ Do
# MAGIC 1. Use **hybrid search** for better recall
# MAGIC 2. Add **reranking** for critical / legal applications
# MAGIC 3. **Filter by metadata** (year, document_type) to narrow results
# MAGIC 4. Monitor **index sync** status after data changes
# MAGIC 5. Use an appropriate **num_results** (5–10 for most cases)
# MAGIC 6. Include relevant **metadata columns** in results
# MAGIC
# MAGIC ### ❌ Don't
# MAGIC 1. Retrieve too many results (increases latency)
# MAGIC 2. Ignore index sync status
# MAGIC 3. Use only semantic search for exact regulation numbers
# MAGIC 4. Forget to handle empty results
# MAGIC 5. Over-rely on similarity scores alone

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook we:
# MAGIC 1. ✅ Created a Vector Search endpoint and Delta Sync index
# MAGIC 2. ✅ Ran similarity search over EU legislation chunks
# MAGIC 3. ✅ Explored metadata filtering
# MAGIC 4. ✅ Compared semantic, hybrid, and reranked search
# MAGIC 5. ✅ Reviewed best practices and index monitoring
# MAGIC
# MAGIC **Next steps**: Set up Genie Space for metadata exploration, and create the
# MAGIC data processing pipeline as a DABs job.

# COMMAND ----------
