# Databricks notebook source
# MAGIC %md
# MAGIC # Data Processing Pipeline
# MAGIC
# MAGIC Scheduled job that keeps the EU policy knowledge base up to date.
# MAGIC
# MAGIC Pipeline steps:
# MAGIC 1. Parse any unprocessed PDFs with AI Parse Documents
# MAGIC 2. Extract, clean, and store chunks with metadata
# MAGIC 3. Sync the Vector Search index so new embeddings are available

# COMMAND ----------

from loguru import logger
from pyspark.sql import SparkSession

from eu_policy_agent.config import get_env, load_config
from eu_policy_agent.data_processor import DataProcessor
from eu_policy_agent.vector_search import VectorSearchManager

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

env = get_env(spark)
cfg = load_config("../../project_config.yml", env=env)

logger.info("Configuration loaded:")
logger.info(f"  Environment: {env}")
logger.info(f"  Catalog    : {cfg.catalog}")
logger.info(f"  Schema     : {cfg.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Process New Documents

# COMMAND ----------

processor = DataProcessor(spark=spark, config=cfg)
processor.process_and_save()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Sync Vector Search Index

# COMMAND ----------

vs_manager = VectorSearchManager(config=cfg)
vs_manager.sync_index()

logger.info("✓ Data processing pipeline complete!")

# COMMAND ----------
