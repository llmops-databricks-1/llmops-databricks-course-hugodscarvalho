"""Unit tests for eu_policy_agent.evaluation.

Tests cover the scorer logic directly (bypassing the @mlflow.genai.scorer
decorator, which is an MLflow detail that requires a live environment).

We test:
- _extract_text: normalise various output formats
- _word_count_under_500: boundary conditions
- _mentions_eu_legislation: recognises EU act names
- _length_score: scoring curve
- create_eval_data_from_file: parsing, skipping blanks/comments
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eu_policy_agent.evaluation import (
    _extract_text,
    _length_score,
    _mentions_eu_legislation,
    _word_count_under_500,
    create_eval_data_from_file,
)

# _extract_text


class TestExtractText:
    def test_plain_string(self) -> None:
        assert _extract_text("hello world") == "hello world"

    def test_list_of_strings(self) -> None:
        assert _extract_text(["first", "second"]) == "first"

    def test_list_of_dicts_with_text_key(self) -> None:
        assert _extract_text([{"text": "GDPR content"}]) == "GDPR content"

    def test_list_of_dicts_without_text_key(self) -> None:
        result = _extract_text([{"other": "value"}])
        assert "other" in result  # falls back to str()

    def test_non_string_non_list(self) -> None:
        result = _extract_text(42)  # type: ignore[arg-type]
        assert result == "42"

    def test_empty_list_falls_back_to_str(self) -> None:
        result = _extract_text([])
        assert isinstance(result, str)


# _word_count_under_500


class TestWordCountUnder500:
    def test_short_response_passes(self) -> None:
        text = "GDPR Article 5 lists data minimisation principles."
        assert _word_count_under_500(text) is True

    def test_exactly_499_words_passes(self) -> None:
        text = " ".join(["word"] * 499)
        assert _word_count_under_500(text) is True

    def test_exactly_500_words_fails(self) -> None:
        text = " ".join(["word"] * 500)
        assert _word_count_under_500(text) is False

    def test_very_long_response_fails(self) -> None:
        text = " ".join(["word"] * 1000)
        assert _word_count_under_500(text) is False

    def test_empty_string_passes(self) -> None:
        assert _word_count_under_500("") is True


# _mentions_eu_legislation


class TestMentionsEuLegislation:
    @pytest.mark.parametrize(
        "text",
        [
            "The AI Act introduces a risk-based framework.",
            "Under GDPR Article 20 data portability applies.",
            "The Digital Services Act requires transparency.",
            "DSA platforms must assess systemic risk.",
            "NIS2 applies to essential entities.",
            "The Data Act gives users access rights.",
            "Under the regulation providers must comply.",
            "This directive harmonises rules across the EU.",
        ],
    )
    def test_recognises_eu_acts(self, text: str) -> None:
        assert _mentions_eu_legislation(text) is True

    def test_unrelated_text_fails(self) -> None:
        text = "The weather today is sunny and warm."
        assert _mentions_eu_legislation(text) is False

    def test_case_insensitive_gdpr_upper(self) -> None:
        assert _mentions_eu_legislation("GDPR is important") is True

    def test_case_insensitive_gdpr_lower(self) -> None:
        assert _mentions_eu_legislation("gdpr is important") is True


# _length_score


class TestLengthScore:
    def test_ideal_range_scores_one(self) -> None:
        text = " ".join(["word"] * 150)  # inside 80–400
        score = _length_score(text)
        assert score == pytest.approx(1.0)

    def test_at_lower_boundary_scores_one(self) -> None:
        text = " ".join(["word"] * 80)
        assert _length_score(text) == pytest.approx(1.0)

    def test_at_upper_boundary_scores_one(self) -> None:
        text = " ".join(["word"] * 400)
        assert _length_score(text) == pytest.approx(1.0)

    def test_too_short_penalised(self) -> None:
        text = " ".join(["word"] * 40)  # 40 words, ideal_min=80
        score = _length_score(text)
        assert 0 < score < 1.0

    def test_zero_words_zero_score(self) -> None:
        assert _length_score("") == pytest.approx(0.0)

    def test_very_long_response_scores_near_zero(self) -> None:
        text = " ".join(["word"] * 2000)
        score = _length_score(text)
        assert score == pytest.approx(0.0)


# create_eval_data_from_file


class TestCreateEvalDataFromFile:
    def test_loads_questions(self, tmp_path: Path) -> None:
        f = tmp_path / "eval.txt"
        f.write_text("What is GDPR?\nWhat is the AI Act?\n")

        data = create_eval_data_from_file(str(f))

        assert len(data) == 2
        assert data[0] == {"inputs": {"question": "What is GDPR?"}}
        assert data[1] == {"inputs": {"question": "What is the AI Act?"}}

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "eval.txt"
        f.write_text("Q1\n\n\nQ2\n")

        data = create_eval_data_from_file(str(f))

        assert len(data) == 2

    def test_skips_comment_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "eval.txt"
        f.write_text("# This is a comment\nReal question\n# Another comment\n")

        data = create_eval_data_from_file(str(f))

        assert len(data) == 1
        assert data[0]["inputs"]["question"] == "Real question"

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        f = tmp_path / "eval.txt"
        f.write_text("")

        data = create_eval_data_from_file(str(f))

        assert data == []

    def test_strips_whitespace_from_questions(self, tmp_path: Path) -> None:
        f = tmp_path / "eval.txt"
        f.write_text("  What about NIS2?  \n")

        data = create_eval_data_from_file(str(f))

        assert data[0]["inputs"]["question"] == "What about NIS2?"
