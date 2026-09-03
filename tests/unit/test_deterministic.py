"""Deterministic evaluators.

json_match carries the most weight in the whole design: it is the signal a
promotion gate can trust precisely because a judge cannot influence it
(plan/001 section 3.2.1). Every option below exists because of a real
production mismatch that would otherwise be reported as the model getting
something wrong.
"""

from __future__ import annotations

from typing import Any

import pytest

from evalloop.contracts import EvalContext, EvaluatorSpec, Trace
from evalloop.evaluate import ExactMatchEvaluator, JsonMatchEvaluator

CTX = EvalContext(run_id="r1")


def _trace(**overrides: Any) -> Trace:
    payload: dict[str, Any] = {
        "trace_id": "t1",
        "input": {"user_request": "refund please"},
        "output": {"text": "Sure, refunded."},
        "ground_truth": {},
    }
    payload.update(overrides)
    return Trace.model_validate(payload)


def _exact(**options: Any) -> ExactMatchEvaluator:
    return ExactMatchEvaluator(
        EvaluatorSpec(
            id="reply_match",
            type="exact_match",
            actual="output.text",
            expected="ground_truth.expected_response",
            options=options,
        )
    )


def _json(**options: Any) -> JsonMatchEvaluator:
    return JsonMatchEvaluator(
        EvaluatorSpec(
            id="tool_call_match",
            type="json_match",
            actual="output.tool_calls",
            expected="ground_truth.tool_calls",
            options=options,
        )
    )


# --- exact match ---


def test_identical_strings_pass() -> None:
    trace = _trace(ground_truth={"expected_response": "Sure, refunded."})
    result = _exact().evaluate(trace, CTX)
    assert result.passed is True
    assert result.score == 1.0


def test_different_strings_fail_and_say_how() -> None:
    trace = _trace(ground_truth={"expected_response": "No refund after 30 days."})
    result = _exact().evaluate(trace, CTX)
    assert result.passed is False
    assert "expected" in (result.explanation or "")


@pytest.mark.parametrize(
    ("options", "actual", "expected", "passed"),
    [
        ({}, "  spaced  ", "spaced", True),  # strip is on by default
        ({"strip": False}, "  spaced  ", "spaced", False),
        ({"lower": True}, "REFUNDED", "refunded", True),
        ({}, "REFUNDED", "refunded", False),
        ({"collapse_whitespace": True}, "two   words", "two words", True),
        ({}, "two   words", "two words", False),
    ],
)
def test_normalization_options(
    options: dict[str, Any], actual: str, expected: str, passed: bool
) -> None:
    trace = _trace(output={"text": actual}, ground_truth={"expected_response": expected})
    assert _exact(**options).evaluate(trace, CTX).passed is passed


def test_missing_ground_truth_is_not_applicable_not_a_failure() -> None:
    """Calling it a failure would invent a verdict the data does not support,
    and P4 could then compile it into training data."""
    result = _exact().evaluate(_trace(), CTX)
    assert result.passed is None
    assert result.score is None
    assert result.ground_truth is None
    assert "not applicable" in (result.explanation or "")
    assert not result.is_failure


def test_not_applicable_still_records_the_prediction() -> None:
    """So a later run that acquires ground truth can be compared against what
    the model actually said, without re-reading the trace."""
    result = _exact().evaluate(_trace(), CTX)
    assert result.normalized_prediction == "Sure, refunded."


def test_an_unresolvable_actual_path_is_not_applicable() -> None:
    """The check is misconfigured, or the trace shape is different from what it
    expects. Either way it has nothing to look at."""
    evaluator = ExactMatchEvaluator(
        EvaluatorSpec(
            id="e",
            type="exact_match",
            actual="output.nonexistent",
            expected="ground_truth.expected_response",
        )
    )
    result = evaluator.evaluate(_trace(ground_truth={"expected_response": "x"}), CTX)
    assert result.passed is None
    assert "no value at" in (result.explanation or "")


def test_the_model_saying_nothing_is_a_failure_not_a_gap() -> None:
    """`output.text` is None because the model produced no text. That is a real
    answer and a real failure - distinct from the path not resolving at all."""
    trace = _trace(output={}, ground_truth={"expected_response": "Sure, refunded."})
    result = _exact().evaluate(trace, CTX)
    assert result.passed is False
    assert result.is_failure


def test_no_expected_path_configured_is_not_applicable() -> None:
    evaluator = ExactMatchEvaluator(EvaluatorSpec(id="e", type="exact_match", actual="output.text"))
    result = evaluator.evaluate(_trace(), CTX)
    assert result.passed is None
    assert "no 'expected' path" in (result.explanation or "")


def test_version_hash_changes_with_options() -> None:
    """Otherwise a metric moving and the check changing underneath you are
    indistinguishable."""
    assert _exact().version_hash() != _exact(lower=True).version_hash()


def test_version_hash_is_stable() -> None:
    assert _exact(lower=True).version_hash() == _exact(lower=True).version_hash()


# --- json match: the tool-call check ---

CALL = {"name": "issue_refund", "arguments": {"order_id": "ORD-1", "amount": 79.99}}


def test_matching_tool_calls_pass() -> None:
    trace = _trace(output={"tool_calls": [CALL]}, ground_truth={"tool_calls": [CALL]})
    assert _json().evaluate(trace, CTX).passed is True


def test_wrong_tool_fails_and_names_the_field() -> None:
    """ "tool_calls[0].name: expected X, got Y" is actionable. "did not match"
    sends someone reading two blobs of JSON side by side."""
    trace = _trace(
        output={"tool_calls": [CALL]},
        ground_truth={"tool_calls": [{"name": "open_warranty_claim", "arguments": {}}]},
    )
    result = _json().evaluate(trace, CTX)
    assert result.passed is False
    assert "name" in (result.explanation or "")


def test_argument_order_never_matters() -> None:
    """Dict ordering is an artefact of serialization, not of behaviour."""
    trace = _trace(
        output={"tool_calls": [{"name": "f", "arguments": {"a": 1, "b": 2}}]},
        ground_truth={"tool_calls": [{"name": "f", "arguments": {"b": 2, "a": 1}}]},
    )
    assert _json().evaluate(trace, CTX).passed is True


def test_call_order_matters_unless_ignore_order() -> None:
    """Two calls in a different sequence may or may not be equivalent, so it is
    the suite author's call rather than a default."""
    first = {"name": "cancel_order", "arguments": {}}
    second = {"name": "notify_customer", "arguments": {}}
    trace = _trace(
        output={"tool_calls": [first, second]},
        ground_truth={"tool_calls": [second, first]},
    )
    assert _json().evaluate(trace, CTX).passed is False
    assert _json(ignore_order=True).evaluate(trace, CTX).passed is True


def test_numeric_ids_can_be_coerced() -> None:
    """The database returns order_id 42; the model emits "42". Same order."""
    trace = _trace(
        output={"tool_calls": [{"name": "f", "arguments": {"order_id": "42"}}]},
        ground_truth={"tool_calls": [{"name": "f", "arguments": {"order_id": 42}}]},
    )
    assert _json().evaluate(trace, CTX).passed is False
    assert _json(coerce_types=True).evaluate(trace, CTX).passed is True


def test_booleans_are_not_coerced_to_strings() -> None:
    """`True` is 1 in Python. Coercing it would make `force: true` match
    `force: "1"`, which is a different request."""
    trace = _trace(
        output={"tool_calls": [{"name": "f", "arguments": {"force": True}}]},
        ground_truth={"tool_calls": [{"name": "f", "arguments": {"force": "1"}}]},
    )
    assert _json(coerce_types=True).evaluate(trace, CTX).passed is False


def test_null_and_absent_are_the_same_by_default() -> None:
    trace = _trace(
        output={"tool_calls": [{"name": "f", "arguments": {"note": None, "id": "1"}}]},
        ground_truth={"tool_calls": [{"name": "f", "arguments": {"id": "1"}}]},
    )
    assert _json().evaluate(trace, CTX).passed is True
    assert _json(treat_null_as_missing=False).evaluate(trace, CTX).passed is False


def test_extra_arguments_fail_unless_allowed() -> None:
    """A model passing an unrequested argument has changed the request, so the
    default is strict."""
    trace = _trace(
        output={"tool_calls": [{"name": "f", "arguments": {"id": "1", "dry_run": True}}]},
        ground_truth={"tool_calls": [{"name": "f", "arguments": {"id": "1"}}]},
    )
    result = _json().evaluate(trace, CTX)
    assert result.passed is False
    assert "unexpected" in (result.explanation or "")
    assert _json(allow_extra_arguments=True).evaluate(trace, CTX).passed is True


def test_ignore_paths_drops_fields_that_can_never_match() -> None:
    trace = _trace(
        output={"tool_calls": [{"name": "f", "arguments": {"id": "1", "request_id": "abc"}}]},
        ground_truth={"tool_calls": [{"name": "f", "arguments": {"id": "1", "request_id": "xyz"}}]},
    )
    assert _json().evaluate(trace, CTX).passed is False
    assert _json(ignore_paths=["arguments.request_id"]).evaluate(trace, CTX).passed is True


def test_a_typod_ignore_path_fails_at_construction() -> None:
    """Not at trace 4,000 of 10,000."""
    with pytest.raises(ValueError):
        _json(ignore_paths=["arguments..request_id"])


def test_wrong_number_of_calls_says_how_many() -> None:
    trace = _trace(
        output={"tool_calls": [CALL, CALL]},
        ground_truth={"tool_calls": [CALL]},
    )
    result = _json().evaluate(trace, CTX)
    assert result.passed is False
    assert "1 item" in (result.explanation or "")


def test_empty_expected_matches_empty_actual() -> None:
    """ "the agent should not have called anything" is a real requirement, and
    the most common correct answer for an out-of-policy request."""
    trace = _trace(output={"tool_calls": []}, ground_truth={"tool_calls": []})
    assert _json().evaluate(trace, CTX).passed is True


def test_calling_a_tool_when_none_was_expected_fails() -> None:
    trace = _trace(output={"tool_calls": [CALL]}, ground_truth={"tool_calls": []})
    assert _json().evaluate(trace, CTX).passed is False


def test_missing_ground_truth_is_not_applicable() -> None:
    trace = _trace(output={"tool_calls": [CALL]})
    result = _json().evaluate(trace, CTX)
    assert result.passed is None
    assert not result.is_failure
    assert result.normalized_prediction is not None


def test_a_recorded_null_ground_truth_is_a_real_claim() -> None:
    """`tool_calls: null` means "no tool should have been called". That is an
    answer, and it must be compared, not treated as absent."""
    trace = _trace(output={"tool_calls": [CALL]}, ground_truth={"tool_calls": None})
    result = _json().evaluate(trace, CTX)
    assert result.passed is False


def test_type_mismatch_is_reported_clearly() -> None:
    trace = _trace(
        output={"tool_calls": [{"name": "f", "arguments": {"id": "1"}}]},
        ground_truth={"tool_calls": {"name": "f"}},
    )
    result = _json().evaluate(trace, CTX)
    assert result.passed is False
    assert "expected an object" in (result.explanation or "")


def test_empty_arguments_and_absent_arguments_are_the_same() -> None:
    """ToolCall.arguments defaults to {}, so a model output always carries the
    key while hand-written ground truth routinely omits it. Without this, every
    such pair reported a false failure on a call that matched perfectly - which
    is exactly what this evaluator exists to avoid."""
    trace = _trace(
        output={"tool_calls": [{"name": "cancel_order"}]},
        ground_truth={"tool_calls": [{"name": "cancel_order"}]},
    )
    assert _json().evaluate(trace, CTX).passed is True


def test_empty_versus_populated_arguments_still_fails() -> None:
    """The leniency must not swallow a real difference."""
    trace = _trace(
        output={"tool_calls": [{"name": "f", "arguments": {"id": "1"}}]},
        ground_truth={"tool_calls": [{"name": "f", "arguments": {}}]},
    )
    assert _json().evaluate(trace, CTX).passed is False

    reversed_trace = _trace(
        output={"tool_calls": [{"name": "f", "arguments": {}}]},
        ground_truth={"tool_calls": [{"name": "f", "arguments": {"id": "1"}}]},
    )
    assert _json().evaluate(reversed_trace, CTX).passed is False


def test_the_leniency_can_be_turned_off() -> None:
    trace = _trace(
        output={"tool_calls": [{"name": "cancel_order"}]},
        ground_truth={"tool_calls": [{"name": "cancel_order"}]},
    )
    assert _json(treat_empty_as_missing=False).evaluate(trace, CTX).passed is False


def test_json_match_version_hash_is_stable_and_option_sensitive() -> None:
    assert _json().version_hash() == _json().version_hash()
    assert _json().version_hash() != _json(ignore_order=True).version_hash()


def test_json_match_unresolvable_actual_path_is_not_applicable() -> None:
    evaluator = JsonMatchEvaluator(
        EvaluatorSpec(
            id="e",
            type="json_match",
            actual="output.nonexistent",
            expected="ground_truth.tool_calls",
        )
    )
    result = evaluator.evaluate(_trace(ground_truth={"tool_calls": []}), CTX)
    assert result.passed is None
    assert "no value at" in (result.explanation or "")


def test_expecting_a_list_but_getting_an_object_is_reported() -> None:
    trace = _trace(
        output={"text": "not a list"},
        ground_truth={"expected": ["a"]},
    )
    evaluator = JsonMatchEvaluator(
        EvaluatorSpec(
            id="e", type="json_match", actual="output.text", expected="ground_truth.expected"
        )
    )
    result = evaluator.evaluate(trace, CTX)
    assert result.passed is False
    assert "expected a list" in (result.explanation or "")


def test_ignored_marker_is_readable_in_debug_output() -> None:
    from evalloop.evaluate.deterministic.json_match import _IGNORED

    assert repr(_IGNORED) == "IGNORED"


def test_json_match_with_no_expected_path_configured() -> None:
    """A structural check with nothing to compare against is misconfigured, not
    failing. Saying so beats a passing row that compared nothing."""
    evaluator = JsonMatchEvaluator(
        EvaluatorSpec(id="e", type="json_match", actual="output.tool_calls")
    )
    result = evaluator.evaluate(_trace(output={"tool_calls": [CALL]}), CTX)
    assert result.passed is None
    assert "no 'expected' path" in (result.explanation or "")
