# Databricks notebook source
"""
Week 3 — Simple RAG (Retrieval-Augmented Generation)
=====================================================
Topics covered:
  - What is RAG and when to use it vs. full agent tool-calling
  - Direct Vector Search retrieval over EU policy chunks
  - Building a grounded prompt from retrieved context
  - Single-turn RAG query function
  - Multi-turn RAG with conversation history (SimpleRAG class)
  - RAG vs. agent tool-calling: trade-offs

This notebook is complementary to 3.1_agent_tool_calling.py.
Use it to understand the retrieval-first, deterministic approach
before moving to the full agentic loop.
"""

# COMMAND ----------

import os

from databricks.sdk import WorkspaceClient
from databricks.vector_search.client import VectorSearchClient
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pyspark.sql import SparkSession

from eu_policy_agent.config import get_env, load_config

# COMMAND ----------
# Environment setup

if "DATABRICKS_RUNTIME_VERSION" not in os.environ:
    load_dotenv()
    profile = os.environ["PROFILE"]

spark = SparkSession.builder.getOrCreate()
env = get_env(spark)
cfg = load_config("../project_config.yml", env)

w = WorkspaceClient()

# OpenAI-compatible client pointed at Databricks Model Serving
llm_client = OpenAI(
    api_key=w.tokens.create(lifetime_seconds=1200).token_value,
    base_url=f"{w.config.host}/serving-endpoints",
)

# Vector Search client
vsc = VectorSearchClient(
    workspace_url=w.config.host,
    personal_access_token=w.tokens.create(lifetime_seconds=1200).token_value,
)

logger.info(f"Environment   : {env}")
logger.info(f"Catalog/schema: {cfg.catalog}.{cfg.schema}")
logger.info(f"LLM endpoint  : {cfg.llm_endpoint}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 — What is RAG?
# MAGIC
# MAGIC **RAG (Retrieval-Augmented Generation)** grounds LLM responses in your own
# MAGIC data by injecting retrieved context directly into the prompt.
# MAGIC
# MAGIC ```
# MAGIC User Question
# MAGIC     ↓
# MAGIC Vector Search  →  retrieve top-k chunks from eu_policy_chunks
# MAGIC     ↓
# MAGIC Build Prompt   →  question + retrieved context
# MAGIC     ↓
# MAGIC LLM            →  generate a grounded answer
# MAGIC     ↓
# MAGIC Response (with citations)
# MAGIC ```
# MAGIC
# MAGIC ### RAG vs. Agent tool-calling
# MAGIC
# MAGIC | Aspect | Simple RAG | Agent tool-calling |
# MAGIC |---|---|---|
# MAGIC | Retrieval | Always (deterministic) | On LLM decision |
# MAGIC | Flow | Fixed, one-shot | Dynamic, multi-step |
# MAGIC | Latency | Lower | Higher |
# MAGIC | Flexibility | Lower | Higher |
# MAGIC | Best for | Single-intent Q&A | Complex, multi-step queries |
# MAGIC
# MAGIC **Rule of thumb:** start with Simple RAG.  Upgrade to the full agent loop
# MAGIC only when deterministic retrieval is insufficient.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — Vector Search Retrieval

# COMMAND ----------

INDEX_NAME = f"{cfg.catalog}.{cfg.schema}.eu_policy_index"


def retrieve_chunks(query: str, num_results: int = 5) -> list[dict]:
    """Retrieve relevant EU legislation chunks from Vector Search.

    Uses hybrid search (keyword + semantic) for best recall over
    the eu_policy_chunks table.

    Args:
        query: Natural language search query.
        num_results: Number of chunks to return.

    Returns:
        List of chunk dicts with keys: chunk_text, regulation, article, chunk_id.
    """
    index = vsc.get_index(index_name=INDEX_NAME)
    results = index.similarity_search(
        query_text=query,
        columns=["chunk_text", "regulation", "article", "chunk_id"],
        num_results=num_results,
        query_type="hybrid",
    )

    chunks = []
    if results and "result" in results:
        for row in results["result"].get("data_array", []):
            chunks.append(
                {
                    "chunk_text": row[0],
                    "regulation": row[1],
                    "article": row[2],
                    "chunk_id": row[3],
                }
            )
    return chunks


# COMMAND ----------
# Smoke-test retrieval

query = "What are the obligations of high-risk AI system providers?"
chunks = retrieve_chunks(query, num_results=3)

logger.info(f"Retrieved {len(chunks)} chunk(s) for: '{query}'")
for i, chunk in enumerate(chunks, 1):
    logger.info(
        f"\n{i}. [{chunk['regulation']} — {chunk['article']}]"
        f"\n   {chunk['chunk_text'][:200]}…"
    )

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — Building a Grounded Prompt

# COMMAND ----------


def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    """Build a system+user prompt pair that injects retrieved context.

    The context block cites the regulation and article for each chunk so
    the LLM can produce attribution-ready answers.

    Args:
        question: The user's natural language question.
        chunks: Retrieved chunks from eu_policy_chunks.

    Returns:
        Formatted prompt string ready for a single-turn completion request.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[{i}] {chunk['regulation']} — {chunk['article']}\n{chunk['chunk_text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return f"""You are an expert on EU digital legislation. Answer the question using ONLY the context below.
Cite the regulation and article (e.g., "EU AI Act, Article 13") when making claims.
If the context does not contain enough information, say so explicitly — do not speculate.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


# COMMAND ----------
# Preview the constructed prompt

test_prompt = build_rag_prompt(query, chunks)
logger.info(f"Prompt length : {len(test_prompt)} characters")
logger.info(f"Preview:\n{test_prompt[:600]}…")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — Single-Turn RAG Query

# COMMAND ----------


def rag_query(question: str, num_docs: int = 5) -> dict:
    """Answer a question about EU legislation using RAG.

    Args:
        question: Natural language question about EU digital regulations.
        num_docs: Number of chunks to retrieve and inject.

    Returns:
        Dict with keys: question, answer, sources.
    """
    chunks = retrieve_chunks(question, num_results=num_docs)
    prompt = build_rag_prompt(question, chunks)

    response = llm_client.chat.completions.create(
        model=cfg.llm_endpoint,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=cfg.model_config.get("max_tokens", 1000)
        if hasattr(cfg, "model_config")
        else 1000,
        temperature=0,  # Deterministic — RAG should not hallucinate
    )

    return {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": [
            {"regulation": c["regulation"], "article": c["article"]} for c in chunks
        ],
    }


# COMMAND ----------
# Test single-turn RAG

result = rag_query(
    "What obligations does GDPR impose on data controllers regarding personal data processing?"
)

logger.info("=" * 80)
logger.info(f"Question: {result['question']}")
logger.info("=" * 80)
logger.info(f"\nAnswer:\n{result['answer']}")
logger.info("\nSources:")
for src in result["sources"]:
    logger.info(f"  • {src['regulation']} — {src['article']}")

# COMMAND ----------
# A few more EU regulation questions

questions = [
    "What does the EU AI Act say about prohibited AI practices?",
    "What are the transparency obligations for Very Large Online Platforms under the DSA?",
    "What cybersecurity measures does NIS2 require from essential entities?",
]

for q in questions:
    r = rag_query(q, num_docs=3)
    logger.info("-" * 60)
    logger.info(f"Q: {q}")
    logger.info(f"A: {r['answer'][:300]}…")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5 — Multi-Turn RAG with Conversation History
# MAGIC
# MAGIC Extend the basic RAG loop to support follow-up questions.
# MAGIC Context is re-retrieved on every turn based on the latest question,
# MAGIC while prior conversation turns are kept in the message history so
# MAGIC the LLM can resolve references (e.g., "what about that article?").

# COMMAND ----------


class SimpleRAG:
    """Stateless RAG system with per-session conversation history.

    Retrieves fresh context on every turn using the current question as
    the search query. Conversation history is held in-memory (not persisted
    — see notebook 3.2 for persistent memory via Lakebase).

    Args:
        llm_endpoint: Databricks model serving endpoint name.
        index_name: Fully-qualified Vector Search index name.
        num_docs: Default number of chunks to retrieve per turn.
    """

    def __init__(
        self,
        llm_endpoint: str,
        index_name: str,
        num_docs: int = 4,
    ) -> None:
        self.llm_endpoint = llm_endpoint
        self.index_name = index_name
        self.num_docs = num_docs
        self.conversation_history: list[dict] = []

        _w = WorkspaceClient()
        self._client = OpenAI(
            api_key=_w.tokens.create(lifetime_seconds=1200).token_value,
            base_url=f"{_w.config.host}/serving-endpoints",
        )
        self._vsc = VectorSearchClient(
            workspace_url=_w.config.host,
            personal_access_token=_w.tokens.create(lifetime_seconds=1200).token_value,
        )

    def _retrieve(self, query: str) -> list[dict]:
        """Retrieve chunks for the given query."""
        index = self._vsc.get_index(index_name=self.index_name)
        results = index.similarity_search(
            query_text=query,
            columns=["chunk_text", "regulation", "article"],
            num_results=self.num_docs,
            query_type="hybrid",
        )
        chunks = []
        if results and "result" in results:
            for row in results["result"].get("data_array", []):
                chunks.append(
                    {"chunk_text": row[0], "regulation": row[1], "article": row[2]}
                )
        return chunks

    def chat(self, question: str) -> str:
        """Ask a question, optionally building on the prior conversation.

        Args:
            question: Current user question.

        Returns:
            Assistant answer string.
        """
        chunks = self._retrieve(question)

        context = "\n\n---\n\n".join(
            f"[{c['regulation']} — {c['article']}]\n{c['chunk_text']}" for c in chunks
        )

        system_message = (
            "You are an expert on EU digital legislation. "
            "Answer using ONLY the provided context. "
            "Cite regulation and article for every claim. "
            "If the context is insufficient, say so.\n\n"
            f"CONTEXT:\n{context}"
        )

        self.conversation_history.append({"role": "user", "content": question})

        messages = [
            {"role": "system", "content": system_message}
        ] + self.conversation_history

        response = self._client.chat.completions.create(
            model=self.llm_endpoint,
            messages=messages,
            max_tokens=800,
            temperature=0,
        )

        answer = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": answer})
        return answer

    def clear(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []


# COMMAND ----------
# Multi-turn RAG demo

rag = SimpleRAG(llm_endpoint=cfg.llm_endpoint, index_name=INDEX_NAME)
logger.info("✓ SimpleRAG initialised")

# COMMAND ----------

q1 = "What risk categories does the EU AI Act define?"
a1 = rag.chat(q1)
logger.info(f"Q: {q1}")
logger.info(f"A: {a1}\n")

# COMMAND ----------

# Follow-up — leverages conversation history
q2 = "What specific obligations apply to providers of systems in the highest risk category?"
a2 = rag.chat(q2)
logger.info(f"Q: {q2}")
logger.info(f"A: {a2}\n")

# COMMAND ----------

# Second follow-up
q3 = "Are there any exemptions to those obligations?"
a3 = rag.chat(q3)
logger.info(f"Q: {q3}")
logger.info(f"A: {a3}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6 — RAG Best Practices
# MAGIC
# MAGIC **Retrieval quality:**
# MAGIC - Use **hybrid search** (keyword + semantic) — better recall than pure
# MAGIC   semantic on regulatory text with precise legal terminology.
# MAGIC - Tune `num_results` — 3–5 chunks is usually right for focused Q&A;
# MAGIC   complex multi-regulation questions may benefit from 7–10.
# MAGIC - Consider a **reranker** (cross-encoder) for high-precision use cases.
# MAGIC
# MAGIC **Prompt design:**
# MAGIC - Instruct the LLM to cite sources — prevents hallucination drift.
# MAGIC - Set `temperature=0` for deterministic, reproducible answers.
# MAGIC - Use `"do not speculate"` language when factual accuracy is critical.
# MAGIC
# MAGIC **When to upgrade to full agent tool-calling:**
# MAGIC - The user's question requires multiple retrieval steps.
# MAGIC - Retrieval decisions depend on the LLM's reasoning (e.g., deciding
# MAGIC   *which* regulation to search first).
# MAGIC - You need to combine retrieval with non-search tools (e.g., Genie SQL).
# MAGIC
# MAGIC → See `3.1_agent_tool_calling.py` for the full agentic pattern.
