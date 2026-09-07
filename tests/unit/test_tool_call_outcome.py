"""Did the call actually succeed?

The whole check turns on one decision: a trace whose calls record no outcome is
`not_applicable`, never a pass. Most products log the invocation and not the
return, so defaulting to pass would report a flawless success rate for every
customer whose data cannot answer the question.
"""

from __future__ import annotations

from typing import Any

import pytest

from evalloop.contracts import EvalContext, EvaluatorSpec, Trace
from evalloop.evaluate import ToolCallOutcomeEvaluator

CTX = EvalContext(run_id="r1")

POLICY_ERROR = "ERROR 409 - POLICY_VIOLATION, order is 45 days old"


def _trace(calls: list[dict[str, Any]]) -> Trace:
    return Trace.model_validate(
        {
            "trace_id": "t1",
            "input": {"user_request": "refund please"},
            "output": {"text": "Sure, I've processed your full refund.", "tool_calls": calls},
        }
    )


def _check(**options: Any) -> ToolCallOutcomeEvaluator:
    return ToolCallOutcomeEvaluator(
        EvaluatorSpec(
            id="tool_call_outcome",
            type="tool_call_outcome",
            actual="output.tool_calls",
            options=options,
        )
    )


# --- the failure it exists to catch ---


def test_a_call_that_errored_fails_the_trace() -> None:
    """Legal call, defensible tool, reply consistent with it - and the money
    never moved. The other three checks all pass on this trace."""
    result = _check().evaluate(
        _trace([{"name": "issue_refund", "arguments": {"order_id": "O1"}, "error": POLICY_ERROR}]),
        CTX,
    )
    assert result.passed is False
    assert "POLICY_VIOLATION" in (result.explanation or "")
    assert result.raw_output is not None
    assert result.raw_output["failures"][0]["tool"] == "issue_refund"


def test_a_null_error_is_absence_not_success() -> None:
    """`error: null` is what a product writes when it never filled the field in.
    Reading it as success would be the same lie as defaulting to pass."""
    result = _check().evaluate(
        _trace(
            [
                {
                    "name": "issue_refund",
                    "arguments": {},
                    "result": {"refund_id": "RF-1"},
                    "error": None,
                }
            ]
        ),
        CTX,
    )
    assert result.passed is None


def test_an_empty_error_string_is_a_recorded_success() -> None:
    """The distinction that makes this check work at all. On a field named
    `error`, `""` says "I looked, there was none" - the product answered the
    question. `null` or an absent key says it never answered."""
    result = _check().evaluate(
        _trace([{"name": "issue_refund", "arguments": {}, "error": ""}]), CTX
    )
    assert result.passed is True


def test_one_failure_among_several_calls_fails_the_trace() -> None:
    result = _check().evaluate(
        _trace(
            [
                {"name": "lookup_order", "arguments": {}, "error": None},
                {"name": "issue_refund", "arguments": {}, "error": POLICY_ERROR},
            ]
        ),
        CTX,
    )
    assert result.passed is False
    assert result.raw_output["recorded"] == 1


# --- the trap ---


def test_calls_with_no_recorded_outcome_are_not_applicable() -> None:
    """The decision this check turns on. Saying `pass` here would invent a
    perfect success rate out of a product that logs invocations only."""
    result = _check().evaluate(
        _trace([{"name": "issue_refund", "arguments": {"order_id": "O1"}}]), CTX
    )
    assert result.passed is None
    assert "no call records an outcome" in (result.explanation or "")


def test_a_trace_with_no_calls_is_not_applicable() -> None:
    assert _check().evaluate(_trace([]), CTX).passed is None


def test_a_missing_path_is_not_applicable() -> None:
    evaluator = ToolCallOutcomeEvaluator(
        EvaluatorSpec(id="x", type="tool_call_outcome", actual="output.nonexistent")
    )
    assert evaluator.evaluate(_trace([]), CTX).passed is None


# --- products that record failure differently ---


def test_a_dotted_error_field_reaches_inside_the_result() -> None:
    """Ingest mapping cannot reach inside a list, so a product encoding status
    in its return needs the path here rather than a remapping step."""
    result = _check(error_field="result.status", failure_values=["error", "failed"]).evaluate(
        _trace([{"name": "issue_refund", "arguments": {}, "result": {"status": "failed"}}]),
        CTX,
    )
    assert result.passed is False


def test_a_success_value_is_not_read_as_a_failure() -> None:
    """Without `failure_values`, the string "ok" is merely present and would
    count as a fault."""
    result = _check(error_field="result.status", failure_values=["error", "failed"]).evaluate(
        _trace([{"name": "issue_refund", "arguments": {}, "result": {"status": "ok"}}]),
        CTX,
    )
    assert result.passed is True


def test_without_failure_values_any_present_value_is_a_failure() -> None:
    result = _check().evaluate(_trace([{"name": "x", "arguments": {}, "error": "boom"}]), CTX)
    assert result.passed is False


def test_a_whitespace_only_error_is_a_recorded_success() -> None:
    """Same as an empty string: the field was written, and it says nothing went
    wrong. Failing on whitespace would be a false failure over formatting."""
    result = _check().evaluate(_trace([{"name": "x", "arguments": {}, "error": "   "}]), CTX)
    assert result.passed is True


# --- configuration ---


def test_an_expected_path_is_refused() -> None:
    """There is no target: the product's own verdict is the answer."""
    with pytest.raises(ValueError, match="takes no 'expected' path"):
        ToolCallOutcomeEvaluator(
            EvaluatorSpec(
                id="x",
                type="tool_call_outcome",
                actual="output.tool_calls",
                expected="ground_truth.tool_calls",
            )
        )


def test_the_version_hash_covers_the_options() -> None:
    assert _check().version_hash() != _check(error_field="result.status").version_hash()
