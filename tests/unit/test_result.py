"""The result contract, and the three states that are easy to conflate:
passed=False, error set, and invalid_output. Only the first is a model failure."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalloop.contracts import EvalResult, JudgeResponse, TokenUsage

BASE = {"trace_id": "t1", "evaluator_id": "exact", "evaluator_version": "v1"}


def test_score_and_passed_are_independently_optional() -> None:
    """set_comparison yields an F1 with no pass/fail; exact_match yields a
    boolean with no meaningful score. Consumers must handle None, not assume 0."""
    f1_only = EvalResult(**BASE, score=0.83)
    bool_only = EvalResult(**BASE, passed=True)
    assert f1_only.passed is None
    assert bool_only.score is None


def test_is_failure_only_for_a_check_that_ran_and_said_no() -> None:
    """An errored or invalid result is a failure of the evaluation, not of the
    model, and must never be compiled into training data as if the model was
    wrong."""
    assert EvalResult(**BASE, passed=False).is_failure
    assert not EvalResult(**BASE, passed=False, error="timeout").is_failure
    assert not EvalResult(**BASE, passed=False, invalid_output=True).is_failure
    assert not EvalResult(**BASE, passed=True).is_failure
    assert not EvalResult(**BASE).is_failure


def test_unpriced_model_records_none_not_zero() -> None:
    """Zero would read as free. None reads as a gap in the ledger, which is what
    an unknown model price actually is."""
    assert TokenUsage(tokens_in=100, tokens_out=20).cost_usd is None


def test_judge_invalid_output_is_distinct_from_transport_error() -> None:
    """A judge that reliably returns malformed JSON is broken. A judge behind a
    flaky network is not. The judgecard reports these separately."""
    malformed = JudgeResponse(raw="not json", judge_config_hash="h1")
    timed_out = JudgeResponse(raw="", judge_config_hash="h1", error="timeout")
    ok = JudgeResponse(raw='{"a":1}', parsed={"a": 1}, judge_config_hash="h1")

    assert malformed.invalid_output
    assert not timed_out.invalid_output
    assert not ok.invalid_output


def test_result_is_immutable() -> None:
    result = EvalResult(**BASE, passed=True)
    with pytest.raises(ValidationError):
        result.passed = False  # type: ignore[misc]


def test_unknown_field_rejected() -> None:
    with pytest.raises(ValidationError):
        EvalResult(**BASE, verdict="good")  # type: ignore[call-arg]
