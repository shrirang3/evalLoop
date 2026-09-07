"""The Markdown rendering of the tool-call report.

A second rendering of the same counted report, so the counting is not retested
here. What is tested is the part that only matters on the way out: table cells
survive free text, and the same report renders byte-identically twice.
"""

from __future__ import annotations

from evalloop.report import render_markdown
from evalloop.report.tool_calls import (
    ContradictionRow,
    FailureRow,
    SelectionRow,
    ToolCallReport,
    ViolationRow,
)


def _report(**overrides: object) -> ToolCallReport:
    report = ToolCallReport(run_id="run-1", traces=14)
    for key, value in overrides.items():
        setattr(report, key, value)
    return report


def test_a_report_with_no_tool_checks_still_renders_a_header() -> None:
    out = render_markdown(_report())
    assert out.startswith("# Tool calls")
    assert "run-1" in out
    assert "## Wrong tool selections" not in out


def test_each_evaluated_check_becomes_a_section() -> None:
    out = render_markdown(
        _report(
            selection_evaluated=True,
            violations_evaluated=True,
            contradictions_evaluated=True,
            failures_evaluated=True,
        )
    )
    for heading in (
        "## Wrong tool selections",
        "## Registry violations",
        "## Reply contradicts the calls",
        "## Calls the tool rejected",
    ):
        assert heading in out


def test_a_clean_check_says_so_rather_than_rendering_an_empty_table() -> None:
    out = render_markdown(_report(violations_evaluated=True, violations_clean=10))
    assert "No illegal calls. 10 trace(s)" in out
    assert "| code |" not in out


def test_selection_rows_render_as_a_table() -> None:
    out = render_markdown(
        _report(
            selection_evaluated=True,
            selection=[SelectionRow("issue_refund", "open_warranty_claim", 4, "sb-0418")],
        )
    )
    assert "| `issue_refund` | `open_warranty_claim` | 4 | sb-0418 |" in out


def test_coverage_is_rendered_beside_every_table() -> None:
    """The same discipline as the terminal: a table without its denominator
    overstates how much of the dataset the check could see."""
    out = render_markdown(_report(failures_evaluated=True, failures_clean=1, failures_skipped=12))
    assert "_succeeded 1 · no outcome recorded 12_" in out


def test_a_pipe_in_free_text_does_not_break_the_table() -> None:
    """Error strings and judge explanations are written elsewhere. An unescaped
    pipe silently splits a row instead of showing up as a bad character."""
    out = render_markdown(
        _report(
            failures_evaluated=True,
            failures=[FailureRow("issue_refund", "ERR | code=409 | denied", 1, "t1")],
        )
    )
    assert r"ERR \| code=409 \| denied" in out


def test_a_newline_in_free_text_is_flattened() -> None:
    out = render_markdown(
        _report(
            contradictions_evaluated=True,
            contradictions=[ContradictionRow("t1", "issue_refund", "says X\nnot Y")],
        )
    )
    assert "| t1 | `issue_refund` | says X not Y |" in out


def test_a_violation_with_no_tools_renders_a_dash() -> None:
    out = render_markdown(
        _report(violations_evaluated=True, violations=[ViolationRow("unnamed_call", (), 1, "t1")])
    )
    assert "| `unnamed_call` | — | 1 | t1 |" in out


def test_the_provenance_footer_is_always_present() -> None:
    """The table is half judge-derived, and a file that travels away from the
    terminal has to carry that with it."""
    out = render_markdown(_report(selection_evaluated=True))
    assert "judge-derived" in out
    assert "objective" in out


def test_rendering_is_deterministic() -> None:
    """No timestamps, so the file can be committed and a diff means the results
    moved rather than the clock."""
    report = _report(
        selection_evaluated=True,
        selection=[SelectionRow("issue_refund", "cancel_order", 2, "t1")],
    )
    assert render_markdown(report) == render_markdown(report)
