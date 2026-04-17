# Databricks notebook source
"""
Week 4 — MLflow Tracing for the EU Policy Agent
=================================================
Topics covered:
  - Why tracing matters for GenAI observability
  - @mlflow.trace decorator and SpanType hierarchy
  - Manual span creation with mlflow.start_span
  - mlflow.update_current_trace for metadata and tags
  - Searching and filtering traces by session, git SHA, etc.
  - Performance analysis across recent traces
  - The EuPolicyAgent tracing architecture

Tracing hierarchy in EuPolicyAgent:
  predict_stream()          → AGENT  (root)
    load_memory()           → RETRIEVER
    call_and_run_tools()    → CHAIN
      call_llm()            → LLM    (per iteration)
      execute_tool()        → TOOL   (per tool call)
    save_memory()           → CHAIN
"""

# COMMAND ----------

import os
import random
from datetime import datetime

import mlflow
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from loguru import logger
from mlflow.entities import SpanType
from mlflow.types.responses import ResponsesAgentRequest
from pyspark.sql import SparkSession

from eu_policy_agent.agent import EuPolicyAgent
from eu_policy_agent.config import get_env, load_config

# COMMAND ----------
# Environment / MLflow setup

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ["PROFILE"]
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)
w = WorkspaceClient()

experiment_path = cfg.experiment_path or "/Shared/eu-policy-agent-dev"
mlflow.set_experiment(experiment_path)

logger.info(f"Environment: {env}")
logger.info(f"MLflow experiment: {experiment_path}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — Why Tracing Matters
# MAGIC
# MAGIC When an agent produces a wrong or low-quality answer, you need to know:
# MAGIC - Did retrieval return the wrong chunks?
# MAGIC - Did the LLM misinterpret the tool result?
# MAGIC - Did memory inject stale context?
# MAGIC - Which tool was called (and was it the right one)?
# MAGIC
# MAGIC Without tracing, the agent is a **black box**.  With tracing, every step
# MAGIC is visible, searchable, and comparable across versions.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — @mlflow.trace Decorator

# COMMAND ----------


@mlflow.trace
def simple_retrieval(query: str) -> list[dict]:
    """Simulated retrieval step — traced automatically."""
    # In production this would call VectorSearchManager.search()
    return [{"text": "GDPR Article 5 lists data processing principles..."}]


# Call it — a trace is created automatically
result = simple_retrieval("GDPR data processing principles")
logger.info(f"Retrieval returned {len(result)} chunk(s)")
logger.info("✓ Trace created — check the MLflow Experiments UI")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — SpanType: Classifying Spans

# COMMAND ----------


@mlflow.trace(span_type=SpanType.RETRIEVER, name="eu_legislation_retrieval")
def eu_legislation_retrieval(query: str, num_results: int = 5) -> list[dict]:
    """A retrieval step typed as RETRIEVER for semantic span classification."""
    logger.info(f"Retrieving top-{num_results} chunks for: {query!r}")
    # Placeholder — real call goes to VectorSearchManager
    return [{"text": f"Result for: {query}", "document_type": "regulation"}]


@mlflow.trace(span_type=SpanType.CHAIN, name="rag_pipeline")
def rag_pipeline(question: str) -> str:
    """A simple RAG pipeline typed as CHAIN."""
    chunks = eu_legislation_retrieval(question)
    context = "\n".join(c["text"] for c in chunks)
    return f"Context: {context}\n\nAnswer: (LLM would answer here)"


response = rag_pipeline("What are GDPR data subject rights?")
logger.info("RAG pipeline response:")
logger.info(response)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — Manual Span Control: mlflow.start_span

# COMMAND ----------


@mlflow.trace(span_type=SpanType.CHAIN, name="legislation_analysis")
def analyse_query(question: str) -> dict:
    """Multi-step analysis with manually created sub-spans."""
    with mlflow.start_span(name="intent_detection", span_type=SpanType.CHAIN) as span:
        # Detect which EU act the question relates to
        lower = question.lower()
        if "gdpr" in lower or "personal data" in lower:
            detected_act = "GDPR"
        elif "ai act" in lower or "artificial intelligence" in lower:
            detected_act = "EU AI Act"
        elif "dsa" in lower or "digital services" in lower:
            detected_act = "DSA"
        elif "dma" in lower or "digital markets" in lower:
            detected_act = "DMA"
        else:
            detected_act = "Unknown"

        span.set_inputs({"question": question})
        span.set_outputs({"detected_act": detected_act})

    with mlflow.start_span(name="retrieval", span_type=SpanType.RETRIEVER) as span:
        # Simulate retrieval
        chunks = [f"Relevant chunk about {detected_act}"]
        span.set_inputs({"query": question, "act": detected_act})
        span.set_outputs({"num_chunks": len(chunks)})

    return {"detected_act": detected_act, "num_chunks": len(chunks)}


result = analyse_query("What obligations does GDPR impose on data processors?")
logger.info(f"Analysis result: {result}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5 — mlflow.update_current_trace: Attach Metadata and Tags

# COMMAND ----------


@mlflow.trace(span_type=SpanType.AGENT)
def traced_agent_call(question: str, session_id: str, request_id: str) -> str:
    """Demonstrate attaching searchable metadata to a trace."""
    mlflow.update_current_trace(
        tags={
            "git_sha": os.getenv("GIT_SHA", "local"),
            "model_serving_endpoint_name": os.getenv(
                "MODEL_SERVING_ENDPOINT_NAME", "local"
            ),
            "model_version": "dev",
            "legislation_domain": "eu_digital_policy",
        },
        metadata={"mlflow.trace.session": session_id},
        client_request_id=request_id,
    )
    # Simulate agent work
    return f"Processed: {question}"


ts = datetime.now().strftime("%Y%m%d-%H%M%S")
demo_session_id = f"s-{ts}-{random.randint(100000, 999999)}"
demo_request_id = f"req-{ts}-{random.randint(100000, 999999)}"

result = traced_agent_call(
    question="What is the scope of the Data Act?",
    session_id=demo_session_id,
    request_id=demo_request_id,
)
logger.info(f"Result: {result}")
logger.info("✓ Tags and metadata attached — filterable in MLflow UI")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6 — Full EuPolicyAgent Trace

# COMMAND ----------

agent = EuPolicyAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    catalog=cfg.catalog,
    schema=cfg.schema,
    genie_space_id=cfg.genie_space_id or None,
    lakebase_project_id=None,  # Stateless for this demo
)

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
session_id = f"s-{ts}-{random.randint(100000, 999999)}"
request_id = f"req-{ts}-{random.randint(100000, 999999)}"

test_request = ResponsesAgentRequest(
    input=[
        {
            "role": "user",
            "content": (
                "Under the EU AI Act, what technical documentation must providers "
                "of high-risk AI systems maintain?"
            ),
        }
    ],
    custom_inputs={"session_id": session_id, "request_id": request_id},
)

logger.info(f"Session ID: {session_id}")
logger.info(f"Request ID: {request_id}")

response = agent.predict(test_request)

logger.info("Agent response:")
logger.info("=" * 80)
if response.output:
    content = response.output[-1].content
    if isinstance(content, list):
        logger.info(" ".join(c.get("text", "") for c in content))
    else:
        logger.info(content)
logger.info("=" * 80)
logger.info("✓ Full trace created — navigate to MLflow UI > Experiments > Traces")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7 — Searching Traces

# COMMAND ----------

# Search by session ID
session_traces_df = mlflow.search_traces(
    filter_string=f"request_metadata.`mlflow.trace.session` = '{session_id}'",
    order_by=["timestamp_ms ASC"],
)

logger.info(f"Traces for session {session_id!r}: {len(session_traces_df)}")
if len(session_traces_df) > 0:
    safe_cols = [
        c
        for c in session_traces_df.columns
        if c not in ("request", "response", "spans", "inputs", "outputs")
    ]
    display(session_traces_df[safe_cols].head())

# COMMAND ----------

# Search by git SHA (useful for deployment regression tracking)
git_sha = os.getenv("GIT_SHA", "local")
sha_traces_df = mlflow.search_traces(
    filter_string=f"tags.`git_sha` = '{git_sha}'",
    order_by=["timestamp_ms DESC"],
    max_results=10,
)
logger.info(f"Traces for git SHA {git_sha!r}: {len(sha_traces_df)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8 — Performance Analysis

# COMMAND ----------

recent_traces_df = mlflow.search_traces(
    order_by=["timestamp_ms DESC"],
    max_results=50,
)

if len(recent_traces_df) > 0:
    logger.info(f"Recent traces: {len(recent_traces_df)}")

    if "execution_time_ms" in recent_traces_df.columns:
        durations = recent_traces_df["execution_time_ms"].dropna()
        if len(durations) > 0:
            logger.info(f"Avg duration: {durations.mean():.0f} ms")
            logger.info(f"p95 duration: {durations.quantile(0.95):.0f} ms")
            logger.info(f"Max duration: {durations.max():.0f} ms")

    if "status" in recent_traces_df.columns:
        logger.info("Status breakdown:")
        for status, count in recent_traces_df["status"].value_counts().items():
            logger.info(f"  {status}: {count}")

    safe_cols = [
        c
        for c in recent_traces_df.columns
        if c not in ("request", "response", "spans", "inputs", "outputs")
    ]
    if safe_cols:
        display(recent_traces_df[safe_cols].head(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9 — Tracing Best Practices
# MAGIC
# MAGIC 1. **Always attach `session_id`** — enables conversation-level debugging
# MAGIC 2. **Always attach `git_sha`** — enables regression tracking per deployment
# MAGIC 3. **Use typed SpanType** — LLM/TOOL/RETRIEVER/CHAIN/AGENT make the
# MAGIC    trace tree semantic and searchable
# MAGIC 4. **Log token usage and model name** in LLM spans — critical for cost
# MAGIC    monitoring and model comparison
# MAGIC 5. **Don't over-trace** — one span per logical operation, not one per line
# MAGIC 6. **Set span inputs/outputs** explicitly for critical steps so they appear
# MAGIC    in the trace detail view
