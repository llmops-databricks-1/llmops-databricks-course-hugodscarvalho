# Databricks notebook source
"""
Week 4 — Agent Evaluation with MLflow GenAI
=============================================
Topics covered:
  - Why evaluation is different for GenAI vs classical ML
  - Three types of scorers: Guidelines (LLM-judge), code-based, numeric judge
  - Domain-specific evaluation for EU legislation QA
  - Running mlflow.genai.evaluate and reading the results
  - Combining multiple scorers for a comprehensive evaluation suite
  - The evaluation→iterate→improve loop

Scorers implemented for EU Policy Agent:
  polite_tone_guideline      — binary: is the tone professional?
  stays_in_scope_guideline   — binary: does the answer stick to EU law?
  cites_regulation_guideline — binary: does the answer cite regulations?
  word_count_check           — boolean: response < 500 words?
  mentions_legislation       — boolean: mentions a known EU act?
  response_length_score      — float 0-1: ideal range 80-400 words
  quality_judge              — int 1-5: overall quality (numeric judge)
"""

# COMMAND ----------

import os
from typing import Literal

import mlflow
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from loguru import logger
from mlflow.genai.judges import make_judge
from pyspark.sql import SparkSession

from eu_policy_agent.agent import EuPolicyAgent
from eu_policy_agent.config import get_env, load_config
from eu_policy_agent.evaluation import (
    cites_regulation_guideline,
    create_eval_data_from_file,
    mentions_legislation,
    polite_tone_guideline,
    response_length_score,
    stays_in_scope_guideline,
    word_count_check,
)

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
# MAGIC ## 1 — Why Evaluation is Different for GenAI
# MAGIC
# MAGIC | Classical ML | GenAI / Agents |
# MAGIC |---|---|
# MAGIC | Accuracy, F1, MAE | Tone, groundedness, scope, quality |
# MAGIC | Single correct answer | Many valid phrasings |
# MAGIC | Static prediction | Multi-step reasoning + tool use |
# MAGIC | Final output only | Intermediate steps also matter |
# MAGIC
# MAGIC For EU legislation QA, "correct" means:
# MAGIC - The answer cites the right regulation and article
# MAGIC - It doesn't invent legal requirements
# MAGIC - It stays within the EU digital policy scope
# MAGIC - It is professional in tone

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 — Guidelines: Binary Pass/Fail LLM Scorers

# COMMAND ----------

logger.info("Loaded Guidelines scorers from eu_policy_agent.evaluation:")
logger.info(f"  1. {polite_tone_guideline.name}: polite and professional tone")
logger.info(f"  2. {stays_in_scope_guideline.name}: stays within EU legislation scope")
logger.info(f"  3. {cites_regulation_guideline.name}: cites specific regulations")

# COMMAND ----------

# Manually construct some test cases to validate the guidelines
tone_test_data = [
    {
        "inputs": {"question": "What does GDPR say about data portability?"},
        "outputs": "Just read the regulation, it's all there.",  # rude
    },
    {
        "inputs": {"question": "What does GDPR say about data portability?"},
        "outputs": (
            "Under GDPR Article 20, data subjects have the right to receive their "
            "personal data in a structured, commonly used, machine-readable format."
        ),
    },
]

tone_results = mlflow.genai.evaluate(
    data=tone_test_data,
    scorers=[polite_tone_guideline],
)

logger.info("Polite tone results:")
display(tone_results.tables["eval_results"])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 — Code-Based Scorers (Deterministic)

# COMMAND ----------

logger.info("Loaded code-based scorers from eu_policy_agent.evaluation:")
logger.info("  1. word_count_check  — True if response < 500 words")
logger.info("  2. mentions_legislation — True if response names an EU act")
logger.info("  3. response_length_score — float 0-1 (ideal: 80-400 words)")

# COMMAND ----------

code_test_data = [
    {
        "inputs": {"question": "What is the GDPR?"},
        "outputs": (
            "The General Data Protection Regulation (GDPR) is an EU regulation "
            "that governs the processing of personal data."
        ),
    },
    {
        "inputs": {"question": "What is the GDPR?"},
        "outputs": "It's a data law. " * 200,  # Excessively long
    },
]

code_results = mlflow.genai.evaluate(
    data=code_test_data,
    scorers=[word_count_check, mentions_legislation, response_length_score],
)

logger.info("Code-based scorer results:")
display(code_results.tables["eval_results"])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 — Numeric Judge (LLM-as-Scorer, 1-5 Scale)

# COMMAND ----------

quality_judge = make_judge(
    name="response_quality",
    instructions=(
        "Evaluate the quality of the response in {{ outputs }} to the EU legislation "
        "question in {{ inputs }}.\n\n"
        "Score from 1 to 5:\n"
        "1 - Completely incorrect or hallucinated\n"
        "2 - Partially correct but missing key regulatory details\n"
        "3 - Correct at a high level but lacking specificity (no article numbers)\n"
        "4 - Accurate with good regulatory references\n"
        "5 - Excellent: precise, cites specific articles, well-structured"
    ),
    model=f"databricks:/{cfg.llm_endpoint}",
    feedback_value_type=int,
)

logger.info(f"Quality judge created: {quality_judge.name} (1-5 scale)")

# COMMAND ----------

# Categorical judge: classify the scope of the answer
scope_judge = make_judge(
    name="scope_classification",
    instructions=(
        "Classify whether the response in {{ outputs }} is 'in_scope', "
        "'partially_in_scope', or 'out_of_scope' for an EU digital legislation "
        "assistant."
    ),
    feedback_value_type=Literal["in_scope", "partially_in_scope", "out_of_scope"],
    model=f"databricks:/{cfg.llm_endpoint}",
)

logger.info(f"Scope judge created: {scope_judge.name}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5 — Comprehensive Evaluation

# COMMAND ----------

all_scorers = [
    polite_tone_guideline,
    stays_in_scope_guideline,
    cites_regulation_guideline,
    word_count_check,
    mentions_legislation,
    response_length_score,
    quality_judge,
    scope_judge,
]

comprehensive_test_data = [
    {
        "inputs": {"question": "What are the GDPR data subject rights?"},
        "outputs": (
            "Under GDPR Chapter III, data subjects have the following rights: "
            "(1) Right of access (Article 15); (2) Right to rectification (Article 16); "
            "(3) Right to erasure / 'right to be forgotten' (Article 17); "
            "(4) Right to restriction of processing (Article 18); "
            "(5) Right to data portability (Article 20); "
            "(6) Right to object (Article 21). "
            "These rights can be restricted in certain circumstances, such as when "
            "exercising them would conflict with freedom of expression or legal claims."
        ),
    },
    {
        "inputs": {"question": "What does the EU AI Act say about risk levels?"},
        "outputs": "I don't know anything about that, try Google.",  # Bad answer
    },
]

logger.info("Running comprehensive evaluation…")
comprehensive_results = mlflow.genai.evaluate(
    data=comprehensive_test_data,
    scorers=all_scorers,
)

logger.info("Comprehensive evaluation complete:")
logger.info(f"Metrics: {comprehensive_results.metrics}")
display(comprehensive_results.tables["eval_results"])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6 — Evaluate Against Real EU Legislation Questions

# COMMAND ----------

agent = EuPolicyAgent(
    llm_endpoint=cfg.llm_endpoint,
    system_prompt=cfg.system_prompt,
    catalog=cfg.catalog,
    schema=cfg.schema,
    genie_space_id=cfg.genie_space_id or None,
    lakebase_project_id=None,  # Stateless evaluation
)


def predict_fn(question: str) -> str:
    """Thin wrapper for mlflow.genai.evaluate."""
    request = {"input": [{"role": "user", "content": question}]}
    result = agent.predict(request)
    if not result.output:
        return ""
    content = result.output[-1].content
    if isinstance(content, list):
        return " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
        )
    return str(content)


# COMMAND ----------

# Load evaluation questions from file
eval_data = create_eval_data_from_file("../eval_inputs.txt")
logger.info(f"Loaded {len(eval_data)} evaluation question(s) from eval_inputs.txt")

# Run evaluation (use a subset of scorers for speed)
production_scorers = [
    polite_tone_guideline,
    stays_in_scope_guideline,
    word_count_check,
    mentions_legislation,
    response_length_score,
    quality_judge,
]

logger.info("Running production evaluation…")
production_results = mlflow.genai.evaluate(
    predict_fn=predict_fn,
    data=eval_data,
    scorers=production_scorers,
)

logger.info("Production evaluation metrics:")
for metric_name, value in production_results.metrics.items():
    logger.info(
        f"  {metric_name}: {value:.3f}"
        if isinstance(value, float)
        else f"  {metric_name}: {value}"
    )

display(production_results.tables["eval_results"])

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7 — Evaluation Best Practices
# MAGIC
# MAGIC 1. **Start small** — 10-20 high-quality questions beat 500 low-quality ones
# MAGIC 2. **Include edge cases** — questions the agent might hallucinate on
# MAGIC 3. **Cover all acts** — AI Act, GDPR, DSA, DMA, NIS2, Data Act, DGA
# MAGIC 4. **Code-based scorers first** — cheap, deterministic, no LLM cost
# MAGIC 5. **LLM judges sparingly** — powerful but slower and non-deterministic
# MAGIC 6. **Track metrics over time** — run evaluation before every deployment
# MAGIC 7. **Never use the same model as generator and judge** — score inflation risk
