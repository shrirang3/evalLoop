"""The wrong-tool rollup.

Rows here are shaped the way the metastore actually returns them, envelope and
all: `repo._jsonable` wraps a scalar or a list as `{"value": ...}` on the way
into a JSON object column. Reading the envelope as the value fails silently -
every prediction looks absent and the table fills with rows claiming nothing was
called and nothing was expected - so the shape is pinned by test.
"""

from __future__ import annotations

from typing import Any

from evalloop.report import build_tool_call_report

# --- row builders, matching what `select()` returns from eval_result ---


def _selection(
    trace_id: str,
    *,
    called: list[str] | None = None,
    judge: str | None = None,
    passed: bool | None = False,
    invalid: bool = False,
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "passed": passed,
        "normalized_prediction": {"value": called} if called is not None else None,
        "ground_truth": {"value": judge} if judge is not None else None,
        "raw_output": None,
        "invalid_output": invalid,
    }


def _registry(
    trace_id: str,
    *,
    violations: list[dict[str, Any]] | None = None,
    passed: bool | None = True,
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "passed": passed,
        "normalized_prediction": None,
        "ground_truth": None,
        "raw_output": {"violations": violations} if violations else None,
        "invalid_output": False,
    }


def _consistency(
    trace_id: str,
    *,
    called: list[str] | None = None,
    passed: bool | None = False,
    explanation: str = "",
    invalid: bool = False,
) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "passed": passed,
        "normalized_prediction": {"value": called} if called is not None else None,
        "ground_truth": None,
        "raw_output": None,
        "invalid_output": invalid,
        "explanation": explanation,
    }


def _report(
    selection: list[dict[str, Any]],
    registry: list[dict[str, Any]],
    consistency: list[dict[str, Any]] | None = None,
) -> Any:
    return build_tool_call_report("run-1", selection, registry, consistency)


# --- selection ---


def test_the_stored_envelope_is_unwrapped() -> None:
    """The regression. Both columns arrive as `{"value": ...}`, not bare."""
    report = _report([_selection("t1", called=["issue_refund"], judge="open_warranty_claim")], [])
    assert report.selection[0].called == "issue_refund"
    assert report.selection[0].judge_says == "open_warranty_claim"


def test_identical_disagreements_collapse_into_one_row() -> None:
    rows = [
        _selection(f"t{i}", called=["issue_refund"], judge="open_warranty_claim") for i in range(4)
    ]
    report = _report(rows, [])
    assert len(report.selection) == 1
    assert report.selection[0].count == 4
    assert report.selection[0].example_trace_id == "t0"


def test_rows_are_ordered_by_frequency_then_name() -> None:
    """Ties break by name so the same data renders the same table twice."""
    rows = [
        _selection("t1", called=["track_order"], judge="lookup_order"),
        _selection("t2", called=["issue_refund"], judge="lookup_order"),
        _selection("t3", called=["issue_refund"], judge="lookup_order"),
        _selection("t4", called=["cancel_order"], judge="lookup_order"),
    ]
    report = _report(rows, [])
    assert [(r.called, r.count) for r in report.selection] == [
        ("issue_refund", 2),
        ("cancel_order", 1),
        ("track_order", 1),
    ]


def test_agreement_is_a_number_not_a_row() -> None:
    report = _report(
        [
            _selection("t1", called=["issue_refund"], judge="issue_refund", passed=True),
            _selection("t2", called=["issue_refund"], judge="open_warranty_claim"),
        ],
        [],
    )
    assert report.selection_agreed == 1
    assert report.wrong_selections == 1


def test_not_applicable_and_invalid_are_counted_apart_from_failures() -> None:
    """Neither is the model getting something wrong, so neither belongs in the
    table - but hiding them would overstate how much of the data was seen."""
    report = _report(
        [
            _selection("t1", passed=None),
            _selection("t2", invalid=True, passed=None),
            _selection("t3", called=["issue_refund"], judge="lookup_order"),
        ],
        [],
    )
    assert report.selection_skipped == 1
    assert report.selection_invalid == 1
    assert report.wrong_selections == 1


def test_calling_nothing_reads_as_none() -> None:
    report = _report([_selection("t1", called=[], judge="lookup_order")], [])
    assert report.selection[0].called == "none"


def test_several_calls_are_joined_in_order() -> None:
    """sb-0421: cancelled correctly, then also refunded."""
    report = _report(
        [_selection("t1", called=["cancel_order", "issue_refund"], judge="cancel_order")], []
    )
    assert report.selection[0].called == "cancel_order,issue_refund"


# --- violations ---


def test_violations_group_by_code_with_the_tools_involved() -> None:
    report = _report(
        [],
        [
            _registry(
                "t1",
                passed=False,
                violations=[{"code": "unregistered_tool", "tool": "refund_now", "call": 0}],
            ),
            _registry(
                "t2",
                passed=False,
                violations=[{"code": "unregistered_tool", "tool": "refund_fast", "call": 0}],
            ),
            _registry(
                "t3",
                passed=False,
                violations=[
                    {"code": "not_permitted_at_node", "tool": "issue_refund", "call": 0},
                    {"code": "invalid_arguments", "tool": "issue_refund", "call": 0},
                ],
            ),
        ],
    )
    by_code = {row.code: row for row in report.violations}
    assert by_code["unregistered_tool"].count == 2
    assert by_code["unregistered_tool"].tools == ("refund_fast", "refund_now")
    assert by_code["unregistered_tool"].example == "t1"
    assert report.illegal_calls == 4


def test_clean_and_skipped_traces_are_counted_separately() -> None:
    report = _report(
        [],
        [
            _registry("t1", passed=True),
            _registry("t2", passed=None),
            _registry("t3", passed=False, violations=[{"code": "unnamed_call", "tool": None}]),
        ],
    )
    assert report.violations_clean == 1
    assert report.violations_skipped == 1


def test_an_advisory_violation_is_reported_on_a_passing_trace() -> None:
    """`duplicate_undeclared` is not a failure, and hiding it would hide the one
    thing the registry author needs to fix."""
    report = _report(
        [],
        [
            _registry(
                "t1",
                passed=True,
                violations=[{"code": "duplicate_undeclared", "tool": "route", "call": 1}],
            )
        ],
    )
    assert report.violations_clean == 1
    assert report.violations[0].code == "duplicate_undeclared"


def test_malformed_raw_output_is_ignored_rather_than_crashing() -> None:
    rows = [
        _registry("t1", passed=True),
        {**_registry("t2", passed=True), "raw_output": {"violations": "not a list"}},
        {**_registry("t3", passed=True), "raw_output": "not a dict"},
    ]
    assert _report([], rows).violations == []


# --- shape ---


def test_traces_counts_the_union_of_both_checks() -> None:
    report = _report([_selection("t1", passed=True)], [_registry("t1"), _registry("t2")])
    assert report.traces == 2


def test_a_suite_with_no_tool_checks_is_empty_not_clean() -> None:
    """ "Found nothing" and "never looked" are different claims."""
    report = _report([], [])
    assert report.is_empty
    assert not report.selection_evaluated
    assert not report.violations_evaluated


def test_a_suite_with_only_one_of_the_two_checks_is_not_empty() -> None:
    report = _report([_selection("t1", passed=True)], [])
    assert not report.is_empty
    assert report.selection_evaluated
    assert not report.violations_evaluated


# --- contradictions ---


def test_contradictions_are_listed_per_trace_not_grouped() -> None:
    """No two contradictions share a string, so there is nothing to group by -
    unlike a tool name or a violation code."""
    report = _report(
        [],
        [],
        [
            _consistency(
                "t2", called=["open_warranty_claim"], explanation="reply says 'full refund'"
            ),
            _consistency("t1", called=[], explanation="reply says 'claim opened'"),
        ],
    )
    assert [row.trace_id for row in report.contradictions] == ["t1", "t2"]  # stable order
    assert report.contradicted == 2
    assert report.contradictions[0].called == "none"


def test_consistent_replies_are_counted_not_listed() -> None:
    report = _report(
        [],
        [],
        [
            _consistency("t1", called=["issue_refund"], passed=True),
            _consistency("t2", called=["issue_refund"], explanation="mismatch"),
        ],
    )
    assert report.contradictions_consistent == 1
    assert report.contradicted == 1


def test_traces_with_no_reply_and_invalid_answers_are_counted_apart() -> None:
    report = _report(
        [],
        [],
        [
            _consistency("t1", passed=None),
            _consistency("t2", invalid=True, passed=None),
        ],
    )
    assert report.contradictions_skipped == 1
    assert report.contradictions_invalid == 1
    assert report.contradicted == 0


def test_a_suite_with_only_the_consistency_check_is_not_empty() -> None:
    report = _report([], [], [_consistency("t1", passed=True)])
    assert not report.is_empty
    assert report.contradictions_evaluated
