# Databricks notebook source
import io
import os
from datetime import datetime

import pypdf
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql.types import ArrayType, BooleanType, IntegerType, StringType, StructField, StructType, TimestampType

from eu_policy_agent.config import get_env, load_config

# COMMAND ----------
# Create Spark session
spark = SparkSession.builder.getOrCreate()

# Load config
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

CATALOG = cfg.catalog
SCHEMA = cfg.schema
TABLE_NAME = "raw_documents"

VOLUME_PATH = cfg.full_volume_path  # /Volumes/dev/eu_policy/legislation

# Create schema if it doesn't exist
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
logger.info(f"Schema {CATALOG}.{SCHEMA} ready")

# COMMAND ----------
# Static metadata for each EU legislation document
# Keyed by document_id (PDF filename without extension)

DOCUMENT_METADATA: dict[str, dict] = {
    "ai_act": {
        "official_title": (
            "Regulation (EU) 2024/1689 of the European Parliament and of the Council "
            "of 13 June 2024 laying down harmonised rules on artificial intelligence"
        ),
        "document_type": "Regulation",
        "regulation_number": "2024/1689",
        "year": 2024,
        "topics": ["AI", "risk management", "transparency", "prohibited AI practices",
                   "high-risk AI systems", "conformity assessment", "general-purpose AI"],
        "official_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
    },
    "gdpr": {
        "official_title": (
            "Regulation (EU) 2016/679 of the European Parliament and of the Council "
            "of 27 April 2016 on the protection of natural persons with regard to the "
            "processing of personal data and on the free movement of such data"
        ),
        "document_type": "Regulation",
        "regulation_number": "2016/679",
        "year": 2016,
        "topics": ["data protection", "privacy", "personal data", "consent",
                   "data subject rights", "data controller", "data processor"],
        "official_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679",
    },
    "digital_markets_act": {
        "official_title": (
            "Regulation (EU) 2022/1925 of the European Parliament and of the Council "
            "of 14 September 2022 on contestable and fair markets in the digital sector"
        ),
        "document_type": "Regulation",
        "regulation_number": "2022/1925",
        "year": 2022,
        "topics": ["digital markets", "gatekeepers", "fair competition",
                   "platform regulation", "interoperability", "self-preferencing"],
        "official_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R1925",
    },
    "digital_services_act": {
        "official_title": (
            "Regulation (EU) 2022/2065 of the European Parliament and of the Council "
            "of 19 October 2022 on a Single Market For Digital Services"
        ),
        "document_type": "Regulation",
        "regulation_number": "2022/2065",
        "year": 2022,
        "topics": ["digital services", "online platforms", "content moderation",
                   "illegal content", "transparency", "very large online platforms"],
        "official_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R2065",
    },
    "nis2_directive": {
        "official_title": (
            "Directive (EU) 2022/2555 of the European Parliament and of the Council "
            "of 14 December 2022 on measures for a high common level of cybersecurity "
            "across the Union"
        ),
        "document_type": "Directive",
        "regulation_number": "2022/2555",
        "year": 2022,
        "topics": ["cybersecurity", "network security", "incident reporting",
                   "critical infrastructure", "essential entities", "supply chain security"],
        "official_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022L2555",
    },
    "data_act": {
        "official_title": (
            "Regulation (EU) 2023/2854 of the European Parliament and of the Council "
            "of 13 December 2023 on harmonised rules on fair access to and use of data"
        ),
        "document_type": "Regulation",
        "regulation_number": "2023/2854",
        "year": 2023,
        "topics": ["data access", "data sharing", "IoT", "cloud switching",
                   "data portability", "connected products"],
        "official_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32023R2854",
    },
    "data_governance_act": {
        "official_title": (
            "Regulation (EU) 2022/868 of the European Parliament and of the Council "
            "of 30 May 2022 on European data governance"
        ),
        "document_type": "Regulation",
        "regulation_number": "2022/868",
        "year": 2022,
        "topics": ["data governance", "data intermediaries", "data altruism",
                   "public sector data", "data spaces", "re-use of protected data"],
        "official_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R0868",
    },
}

# COMMAND ----------
# List PDF files from the Unity Catalog Volume


def list_pdf_files(volume_path: str) -> list[str]:
    """List all PDF files in a Unity Catalog Volume path.

    Tries dbutils.fs.ls first (native Databricks runtime), then falls back to
    the Databricks SDK Files API for local development with databricks-connect.

    Args:
        volume_path: Absolute volume path, e.g. /Volumes/dev/eu_policy/legislation

    Returns:
        Sorted list of absolute file paths ending in .pdf
    """
    try:
        # Native Databricks runtime — dbutils is injected into the notebook global scope
        items = dbutils.fs.ls(volume_path)  # noqa: F821
        return sorted([item.path for item in items if item.name.endswith(".pdf")])
    except NameError:
        pass

    # Local dev with databricks-connect: use the SDK Files REST API
    # (no cluster needed — works directly against Unity Catalog Volumes)
    from databricks.sdk import WorkspaceClient

    w = WorkspaceClient()
    items = w.files.list_directory_contents(volume_path)
    return sorted(
        [
            f"{volume_path}/{item.name}"
            for item in items
            if item.name and item.name.endswith(".pdf")
        ]
    )


pdf_files = list_pdf_files(VOLUME_PATH)
logger.info(f"Found {len(pdf_files)} PDF files in {VOLUME_PATH}:")
for path in pdf_files:
    logger.info(f"  {os.path.basename(path)}")

# COMMAND ----------
# Extract text from each PDF


def _open_pdf_stream(file_path: str):
    """Return a binary stream for a PDF, regardless of where it lives.

    On native Databricks the volume is FUSE-mounted, so plain open() works.
    Locally with databricks-connect the path doesn't exist on disk, so we
    download it via the SDK Files REST API instead.
    """
    try:
        return open(file_path, "rb")
    except FileNotFoundError:
        from databricks.sdk import WorkspaceClient
        response = WorkspaceClient().files.download(file_path)
        return io.BytesIO(response.contents.read())


def get_pdf_page_count(file_path: str) -> int:
    """Return the number of pages in a PDF file.

    Full text extraction is deferred to the chunking notebook.

    Args:
        file_path: Absolute path to the PDF (e.g. /Volumes/...)

    Returns:
        Number of pages, or 0 if the file cannot be read.
    """
    try:
        with _open_pdf_stream(file_path) as f:
            return len(pypdf.PdfReader(f).pages)
    except Exception as e:
        logger.warning(f"Failed to read page count from {file_path}: {e}")
        return 0


def ingest_pdf_documents(pdf_file_paths: list[str]) -> list[dict]:
    """Ingest a list of PDF files from a volume and return document records.

    Each record contains page count plus static metadata. Full text extraction
    is deferred to the chunking notebook.

    Args:
        pdf_file_paths: List of absolute paths to PDF files

    Returns:
        List of document metadata dictionaries
    """
    documents = []
    for file_path in pdf_file_paths:
        filename = os.path.basename(file_path)
        document_id = os.path.splitext(filename)[0]

        # Look up static metadata; fall back gracefully for unknown files
        meta = DOCUMENT_METADATA.get(document_id, {})
        official_title = meta.get(
            "official_title",
            document_id.replace("_", " ").replace("-", " ").title(),
        )
        title = official_title.split(" of the European Parliament")[0].strip()

        logger.info(f"Reading page count from {filename}...")
        num_pages = get_pdf_page_count(file_path)

        documents.append(
            {
                "document_id": document_id,
                "filename": filename,
                "title": title,
                "official_title": official_title,
                "document_type": meta.get("document_type"),
                "regulation_number": meta.get("regulation_number"),
                "year": meta.get("year"),
                "topics": meta.get("topics"),
                "official_url": meta.get("official_url"),
                "volume_path": file_path,
                "num_pages": num_pages,
                "ingestion_timestamp": datetime.now(),
                "processed": None,  # Will be set to True in later notebooks (e.g. chunking/embedding)
            }
        )
        logger.info(f"  -> {num_pages} pages | {meta.get('regulation_number', 'unknown')}")

    return documents


logger.info("Starting EU policy PDF ingestion...")
documents = ingest_pdf_documents(pdf_files)
logger.info(f"Completed ingestion of {len(documents)} documents")

# COMMAND ----------
# Create Delta Table in Unity Catalog
# Store document metadata for downstream processing (chunking, embedding, evaluation).

schema = StructType(
    [
        StructField("document_id", StringType(), False),        # e.g. "ai_act"
        StructField("filename", StringType(), False),           # e.g. "ai_act.pdf"
        StructField("title", StringType(), True),               # Short title (before "of the European Parliament")
        StructField("official_title", StringType(), True),      # Full legal title
        StructField("document_type", StringType(), True),       # "Regulation" or "Directive"
        StructField("regulation_number", StringType(), True),   # e.g. "2024/1689"
        StructField("year", IntegerType(), True),               # Publication year
        StructField("topics", ArrayType(StringType()), True),   # Key topic tags
        StructField("official_url", StringType(), True),        # EUR-Lex URL
        StructField("volume_path", StringType(), True),         # /Volumes/.../ai_act.pdf
        StructField("num_pages", IntegerType(), True),          # PDF page count
        StructField("ingestion_timestamp", TimestampType(), True),
        StructField("processed", BooleanType(), True),          # Set to True once document is chunked/embedded
    ]
)

df = spark.createDataFrame(documents, schema=schema)

table_path = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"

df.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable(table_path)

logger.info(f"Created Delta table: {table_path}")
logger.info(f"Records written: {df.count()}")

# COMMAND ----------
# Verify the Data

docs_df = spark.table(f"{CATALOG}.{SCHEMA}.{TABLE_NAME}")

logger.info(f"Table: {CATALOG}.{SCHEMA}.{TABLE_NAME}")
logger.info(f"Total documents: {docs_df.count()}")
logger.info("Schema:")
docs_df.printSchema()

logger.info("Document overview:")
docs_df.select("document_id", "document_type", "regulation_number", "year", "num_pages") \
    .show(20, truncate=60)

# COMMAND ----------
# Data Statistics

logger.info("Page count by document (largest first):")
docs_df.select("document_id", "regulation_number", "document_type", "num_pages") \
    .orderBy("num_pages", ascending=False) \
    .show(truncate=50)

logger.info("Documents by type:")
docs_df.groupBy("document_type").count().orderBy("count", ascending=False).show()

logger.info("Documents by year:")
docs_df.groupBy("year").count().orderBy("year", ascending=False).show()

logger.info("Total corpus statistics:")
docs_df.selectExpr(
    "count(*) as num_documents",
    "sum(num_pages) as total_pages",
    "avg(num_pages) as avg_pages_per_doc",
).show()

# COMMAND ----------
