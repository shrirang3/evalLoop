"""The eval suite: what runs, and the two properties P4 and P6 depend on."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalloop.contracts import EvalSuite

SUITE = {
    "suite": "support-bot-v1",
    "evaluators": [
        {
            "id": "tool_call_match",
            "type": "json_match",
            "actual": "output.tool_calls",
            "expected": "ground_truth.tool_calls",
            "options": {"ignore_order": True},
        },
        {
            "id": "policy_followed",
            "type": "llm_question",
            "question": "Did the agent follow the refund policy?",
            "inputs": {"transcript": "output.text"},
        },
    ],
}


def test_mixed_deterministic_and_llm_evaluators_parse() -> None:
    suite = EvalSuite.model_validate(SUITE)
    assert len(suite.evaluators) == 2
    assert suite.deterministic_ids == ["tool_call_match"]


def test_llm_question_needs_no_ground_truth() -> None:
    """Without a label the question is still measured and reported - it just
    cannot support an absolute claim (plan/001 section 1)."""
    suite = EvalSuite.model_validate(SUITE)
    llm = suite.evaluators[1]
    assert llm.ground_truth is None


def test_holdout_questions_are_identifiable() -> None:
    """P4 asserts none of these reach a training dataset; P6 scores the gate
    with them. Both need to find them by id."""
    spec = {
        **SUITE,
        "evaluators": [
            *SUITE["evaluators"],  # type: ignore[misc]
            {
                "id": "escalation_correct",
                "type": "llm_question",
                "question": "Should this have been escalated?",
                "holdout": True,
            },
        ],
    }
    suite = EvalSuite.model_validate(spec)
    assert suite.holdout_ids == ["escalation_correct"]


def test_duplicate_evaluator_ids_rejected() -> None:
    """Ids key every result row. Two checks sharing one id silently overwrite
    each other's results."""
    spec = {
        "suite": "dup",
        "evaluators": [
            {"id": "same", "type": "exact_match", "actual": "output.text"},
            {"id": "same", "type": "regex", "actual": "output.text"},
        ],
    }
    with pytest.raises(ValidationError, match="duplicate evaluator id"):
        EvalSuite.model_validate(spec)


def test_empty_suite_rejected() -> None:
    with pytest.raises(ValidationError):
        EvalSuite.model_validate({"suite": "empty", "evaluators": []})


def test_typo_in_evaluator_key_is_an_error() -> None:
    """The reason unknown keys are errors: `expcted` would otherwise be silently
    dropped and the check would compare against nothing, passing everything."""
    spec = {
        "suite": "typo",
        "evaluators": [
            {
                "id": "x",
                "type": "exact_match",
                "actual": "output.text",
                "expcted": "ground_truth.expected_response",
            }
        ],
    }
    with pytest.raises(ValidationError):
        EvalSuite.model_validate(spec)


def test_suite_hash_is_stable_and_content_sensitive() -> None:
    """Two runs with the same suite hash are comparable. Two without it are not,
    however similar the YAML looks."""
    a = EvalSuite.model_validate(SUITE)
    b = EvalSuite.model_validate(SUITE)
    assert a.suite_hash() == b.suite_hash()

    changed = EvalSuite.model_validate(
        {
            **SUITE,
            "evaluators": [
                SUITE["evaluators"][0],  # type: ignore[index]
                {
                    "id": "policy_followed",
                    "type": "llm_question",
                    "question": "Did the agent follow the returns policy?",
                    "inputs": {"transcript": "output.text"},
                },
            ],
        }
    )
    assert a.suite_hash() != changed.suite_hash()


def test_suite_hash_ignores_cosmetic_description() -> None:
    """A description edit does not change what is measured."""
    a = EvalSuite.model_validate(SUITE)
    b = EvalSuite.model_validate({**SUITE, "description": "now with a comment"})
    assert a.suite_hash() == b.suite_hash()
