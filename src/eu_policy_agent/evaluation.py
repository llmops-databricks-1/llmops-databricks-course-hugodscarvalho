"""Evaluation scorers and runner for the EU Policy Agent.

This module provides three categories of scorer for use with
``mlflow.genai.evaluate()``:

1. **Guidelines** (LLM-as-judge, binary pass/fail) — check tone, scope
   adherence, and citation behaviour.
2. **Custom code-based scorers** — cheap, deterministic checks: word count,
   mentions a specific regulation name, response length ratio.
3. **Evaluation runner** — ``evaluate_agent()`` wires everything together
   and returns a ``mlflow.models.EvaluationResult``.

Adapting scorers to your use case
----------------------------------
The guidelines are intentionally domain-specific to EU legislation QA.
To add a new guideline:
    1. Define a ``Guidelines(name=..., guidelines=[...], model=...)`` instance.
    2. Add it to the ``scorers`` list in ``evaluate_agent()``.

To add a new code-based scorer:
    1. Decorate a function with ``@mlflow.genai.scorer``.
    2. Accept ``outputs`` and return ``bool | float``.

See ``notebooks/4.2_evaluation.py`` for interactive experimentation.
"""

from __future__ import annotations

from typing import Any

import mlflow
from loguru import logger
from mlflow.genai.scorers import Guidelines

from eu_policy_agent.agent import EuPolicyAgent
from eu_policy_agent.config import ProjectConfig

# ---------------------------------------------------------------------------
# Guideline scorers (LLM-as-judge, binary pass/fail)
# ---------------------------------------------------------------------------

polite_tone_guideline = Guidelines(
    name="polite_tone",
    guidelines=[
        "The response must use a polite and professional tone throughout.",
        "The response should be helpful and clear without being condescending.",
        "The response must avoid dismissive, rude, or excessively casual language.",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

stays_in_scope_guideline = Guidelines(
    name="stays_in_scope",
    guidelines=[
        "The response must only discuss topics related to EU legislation, regulation, "
        "or digital policy.",
        "If the question is outside the scope of EU legislation, the response should "
        "politely acknowledge this and redirect to relevant EU regulatory topics.",
        "The response must not fabricate legislation, article numbers, or regulatory "
        "requirements that do not exist in the EU legal corpus.",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

cites_regulation_guideline = Guidelines(
    name="cites_regulation",
    guidelines=[
        "Where relevant, the response should cite the specific EU regulation or "
        "directive being discussed (e.g. 'AI Act Article 6', 'GDPR Article 13').",
        "Citations should be accurate and not invented.",
        "If no specific citation is available, the response should acknowledge "
        "uncertainty rather than guessing.",
    ],
    model="databricks:/databricks-gpt-oss-120b",
)

# ---------------------------------------------------------------------------
# Custom code-based scorers (deterministic, no LLM cost)
# ---------------------------------------------------------------------------


def _extract_text(outputs: Any) -> str:
    """Normalise the ``outputs`` argument from mlflow.genai.evaluate."""
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict):
            return first.get("text", str(first))
        return str(first)
    return str(outputs)


# Pure logic functions — independently testable without MLflow
# These are wrapped by the @mlflow.genai.scorer decorators below.


def _word_count_under_500(text: str) -> bool:
    """Return True if ``text`` has fewer than 500 words."""
    return len(text.split()) < 500


def _mentions_eu_legislation(text: str) -> bool:
    """Return True if ``text`` references at least one EU legislative act."""
    lower = text.lower()
    eu_acts = [
        "ai act",
        "gdpr",
        "digital services act",
        "digital markets act",
        "dsa",
        "dma",
        "nis2",
        "data act",
        "data governance act",
        "regulation",
        "directive",
        "article",
    ]
    return any(act in lower for act in eu_acts)


def _length_score(text: str) -> float:
    """Score ``text`` length on a 0–1 scale; ideal range is 80–400 words."""
    word_count = len(text.split())
    ideal_min, ideal_max = 80, 400
    if ideal_min <= word_count <= ideal_max:
        return 1.0
    if word_count < ideal_min:
        return word_count / ideal_min if ideal_min > 0 else 0.0
    return max(0.0, 1.0 - (word_count - ideal_max) / ideal_max)


@mlflow.genai.scorer
def word_count_check(outputs: Any) -> bool:
    """Return True if the response is under 500 words.

    EU legislation answers benefit from conciseness.  Very long responses
    tend to lose the reader and may indicate the agent is over-retrieving.
    """
    return _word_count_under_500(_extract_text(outputs))


@mlflow.genai.scorer
def mentions_legislation(outputs: Any) -> bool:
    """Return True if the response references at least one EU legislative act.

    A grounded answer should name the specific regulation it is relying on.
    """
    return _mentions_eu_legislation(_extract_text(outputs))


@mlflow.genai.scorer
def response_length_score(outputs: Any) -> float:
    """Score response length on a 0–1 scale; ideal range is 80–400 words.

    Responses that are too short likely lack substance; responses that are
    too long are hard to parse.  Returns a linear penalty outside the ideal
    window.
    """
    return _length_score(_extract_text(outputs))


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def evaluate_agent(
    cfg: ProjectConfig,
    eval_inputs_path: str,
    extra_scorers: list | None = None,
) -> mlflow.models.EvaluationResult:
    """Run end-to-end evaluation of ``EuPolicyAgent``.

    Instantiates the agent from ``cfg``, loads evaluation questions from
    ``eval_inputs_path``, and runs ``mlflow.genai.evaluate`` with the
    bundled suite of scorers.

    Args:
        cfg: Resolved ``ProjectConfig`` for the target environment.
        eval_inputs_path: Path to a plain-text file with one evaluation
            question per line.
        extra_scorers: Optional list of additional scorers to append to the
            default suite.

    Returns:
        ``mlflow.models.EvaluationResult`` with per-question scores and
        aggregate metrics.
    """
    agent = EuPolicyAgent(
        llm_endpoint=cfg.llm_endpoint,
        system_prompt=cfg.system_prompt,
        catalog=cfg.catalog,
        schema=cfg.schema,
        genie_space_id=cfg.genie_space_id,
        lakebase_project_id=cfg.lakebase_project_id,
    )

    eval_data = create_eval_data_from_file(eval_inputs_path)
    if not eval_data:
        raise ValueError(f"No evaluation questions found in {eval_inputs_path!r}")

    logger.info(f"Running evaluation on {len(eval_data)} question(s)…")

    def predict_fn(question: str) -> str:
        request = {"input": [{"role": "user", "content": question}]}
        result = agent.predict(request)
        output_items = result.output
        if output_items:
            # Extract text content from the last message item
            last = output_items[-1]
            content = (
                last.content if hasattr(last, "content") else last.get("content", "")
            )
            if isinstance(content, list):
                return " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c) for c in content
                )
            return str(content)
        return ""

    default_scorers = [
        polite_tone_guideline,
        stays_in_scope_guideline,
        cites_regulation_guideline,
        word_count_check,
        mentions_legislation,
        response_length_score,
    ]
    scorers = default_scorers + (extra_scorers or [])

    return mlflow.genai.evaluate(
        predict_fn=predict_fn,
        data=eval_data,
        scorers=scorers,
    )


def create_eval_data_from_file(eval_inputs_path: str) -> list[dict[str, Any]]:
    """Load evaluation questions from a plain-text file.

    Skips blank lines and comment lines (starting with ``#``).

    Args:
        eval_inputs_path: Path to the file.

    Returns:
        List of ``{"inputs": {"question": "<question>"}}`` dicts compatible
        with ``mlflow.genai.evaluate``.
    """
    with open(eval_inputs_path) as f:
        return [
            {"inputs": {"question": line.strip()}}
            for line in f
            if line.strip() and not line.startswith("#")
        ]
