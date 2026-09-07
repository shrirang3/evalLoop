"""The tool-call report as Markdown.

A second rendering of the same counted report, not a second report. The terminal
version stays primary - that is where the number gets read - and this exists for
the times it has to leave the terminal: a pull request, a ticket, a doc someone
who does not have the repo checked out will open.

Deliberately free of timestamps and anything else that moves on its own. Two
runs over the same results produce byte-identical files, so the output can be
committed and diffed, and a diff means the results changed rather than that the
clock did.
"""

from __future__ import annotations

from evalloop.report.tool_calls import ToolCallReport

__all__ = ["render_markdown"]


def render_markdown(report: ToolCallReport) -> str:
    lines: list[str] = [
        "# Tool calls",
        "",
        f"Run `{report.run_id}` · {report.traces} traces · **0 labels**",
        "",
    ]

    if report.selection_evaluated:
        lines += _selection(report)
    if report.violations_evaluated:
        lines += _violations(report)
    if report.contradictions_evaluated:
        lines += _contradictions(report)
    if report.failures_evaluated:
        lines += _failures(report)

    lines += [
        "---",
        "",
        "Selection is judge-derived: relative claims only until the judge is calibrated.",
        "Registry and outcome checks are objective — no model in either.",
        "",
    ]
    return "\n".join(lines)


def _selection(report: ToolCallReport) -> list[str]:
    lines = ["## Wrong tool selections", ""]
    if not report.selection:
        lines += [
            f"No wrong tool selections. The judge agreed on {report.selection_agreed} trace(s).",
            "",
        ]
    else:
        lines += [
            "| called | judge says | n | example |",
            "|---|---|--:|---|",
            *(
                f"| `{_cell(row.called)}` | `{_cell(row.judge_says)}` | {row.count} "
                f"| {_cell(row.example_trace_id or '')} |"
                for row in report.selection
            ),
            "",
        ]
    lines += [
        f"_agreed {report.selection_agreed} · not applicable {report.selection_skipped}"
        f" · invalid answers {report.selection_invalid}_",
        "",
    ]
    return lines


def _violations(report: ToolCallReport) -> list[str]:
    lines = ["## Registry violations", ""]
    if not report.violations:
        lines += [
            f"No illegal calls. {report.violations_clean} trace(s) checked against the registry.",
            "",
        ]
    else:
        lines += [
            "| code | tools | n | example |",
            "|---|---|--:|---|",
            *(
                f"| `{_cell(row.code)}` | {_cell(', '.join(row.tools)) or '—'} | {row.count} "
                f"| {_cell(row.example or '')} |"
                for row in report.violations
            ),
            "",
        ]
    lines += [
        f"_clean {report.violations_clean} · no tool calls {report.violations_skipped}_",
        "",
    ]
    return lines


def _contradictions(report: ToolCallReport) -> list[str]:
    lines = ["## Reply contradicts the calls", ""]
    if not report.contradictions:
        lines += [
            f"No reply contradicted its calls. {report.contradictions_consistent} trace(s) checked.",
            "",
        ]
    else:
        lines += [
            "| trace | called | contradiction |",
            "|---|---|---|",
            *(
                f"| {_cell(row.trace_id)} | `{_cell(row.called)}` | {_cell(row.explanation)} |"
                for row in report.contradictions
            ),
            "",
        ]
    lines += [
        f"_consistent {report.contradictions_consistent} · no reply {report.contradictions_skipped}"
        f" · invalid answers {report.contradictions_invalid}_",
        "",
    ]
    return lines


def _failures(report: ToolCallReport) -> list[str]:
    lines = ["## Calls the tool rejected", ""]
    if not report.failures:
        lines += [
            f"No call failed at runtime. {report.failures_clean} trace(s) recorded an outcome.",
            "",
        ]
    else:
        lines += [
            "| tool | error | n | example |",
            "|---|---|--:|---|",
            *(
                f"| `{_cell(row.tool)}` | {_cell(row.error)} | {row.count} "
                f"| {_cell(row.example or '')} |"
                for row in report.failures
            ),
            "",
        ]
    lines += [
        f"_succeeded {report.failures_clean} · no outcome recorded {report.failures_skipped}_",
        "",
    ]
    return lines


def _cell(value: str) -> str:
    """Make a string safe inside a Markdown table cell.

    Error strings and judge explanations are free text written elsewhere - a
    pipe or a newline in one of them would silently break the table rather than
    show up as a bad character.
    """
    return value.replace("|", "\\|").replace("\n", " ").strip()
