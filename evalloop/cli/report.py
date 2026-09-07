"""`evalloop report tools` - the wrong-tool table.

Reads results a run already wrote. No judge calls, no cost, no database writes,
so it is safe to run repeatedly against the same run id.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from evalloop.report import ToolCallReport, build_tool_call_report, render_markdown
from evalloop.store.db import make_engine, session_scope
from evalloop.store.models import EvalResultRow, EvalRun

__all__ = ["report_app"]

report_app = typer.Typer(help="Roll stored results up into a report.", no_args_is_help=True)

_SELECTION_TYPE = "tool_selection"
_REGISTRY_TYPE = "tool_registry_check"
_CONSISTENCY_TYPE = "text_matches_tools"
_OUTCOME_TYPE = "tool_call_outcome"

_COLUMNS = (
    "trace_id",
    "passed",
    "normalized_prediction",
    "ground_truth",
    "raw_output",
    "invalid_output",
)


@report_app.command("tools")
def report_tools(
    run_id: Annotated[
        str | None,
        typer.Argument(help="Run to report on. Defaults to the most recent run."),
    ] = None,
    selection_id: Annotated[
        str,
        typer.Option("--selection", help="Evaluator id of the tool_selection check."),
    ] = _SELECTION_TYPE,
    registry_id: Annotated[
        str,
        typer.Option("--registry", help="Evaluator id of the tool_registry_check."),
    ] = _REGISTRY_TYPE,
    consistency_id: Annotated[
        str,
        typer.Option("--consistency", help="Evaluator id of the text_matches_tools check."),
    ] = _CONSISTENCY_TYPE,
    outcome_id: Annotated[
        str,
        typer.Option("--outcome", help="Evaluator id of the tool_call_outcome check."),
    ] = _OUTCOME_TYPE,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Also write the report to this path as Markdown."),
    ] = None,
) -> None:
    """Which tools were called wrongly, and which calls were illegal."""
    console = Console()

    with session_scope(make_engine()) as session:
        resolved = (
            run_id
            or session.scalars(
                select(EvalRun.id).order_by(EvalRun.started_at.desc()).limit(1)
            ).first()
        )
        if resolved is None:
            console.print("[red]no runs recorded[/red] — run `evalloop evaluate` first")
            raise typer.Exit(1)
        if session.get(EvalRun, resolved) is None:
            console.print(f"[red]no such run:[/red] {resolved}")
            raise typer.Exit(1)

        report = build_tool_call_report(
            resolved,
            _rows(session, resolved, selection_id),
            _rows(session, resolved, registry_id),
            _rows(session, resolved, consistency_id),
            _rows(session, resolved, outcome_id),
        )

    if report.is_empty:
        console.print(
            f"run [bold]{report.run_id}[/bold] has no tool checks "
            f"(looked for '{selection_id}', '{registry_id}', '{consistency_id}' "
            f"and '{outcome_id}')"
        )
        console.print(
            "[dim]add tool_registry_check and tool_selection to the suite, "
            "and a tools.yaml beside it[/dim]"
        )
        raise typer.Exit(1)

    render(console, report)

    if out is not None:
        # Written in addition to the terminal render, never instead of it. The
        # number gets read here; the file is for the places it has to travel to.
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_markdown(report), encoding="utf-8")
        console.print(f"\n[dim]written to {out}[/dim]")


def _rows(session: Any, run_id: str, evaluator_id: str) -> list[dict[str, Any]]:
    found = session.execute(
        select(*(getattr(EvalResultRow, name) for name in _COLUMNS)).where(
            EvalResultRow.run_id == run_id,
            EvalResultRow.evaluator_id == evaluator_id,
        )
    ).all()
    return [dict(zip(_COLUMNS, row, strict=True)) for row in found]


def render(console: Console, report: ToolCallReport) -> None:
    console.print()
    console.print(
        f"[bold]Tool calls[/bold] — run {report.run_id}, {report.traces} traces, "
        f"[dim]0 labels[/dim]"
    )

    if report.selection_evaluated:
        console.print()
        _render_selection(console, report)
    if report.violations_evaluated:
        console.print()
        _render_violations(console, report)
    if report.contradictions_evaluated:
        console.print()
        _render_contradictions(console, report)
    if report.failures_evaluated:
        console.print()
        _render_failures(console, report)

    console.print()
    console.print(
        "[dim]Selection is judge-derived: relative claims only until the judge is "
        "calibrated.\nSelf-consistency, and the abstention it gates, arrive with "
        "judge-health (P3a).[/dim]"
    )


def _render_selection(console: Console, report: ToolCallReport) -> None:
    if not report.selection:
        console.print(
            f"[green]No wrong tool selections.[/green] "
            f"Judge agreed on {report.selection_agreed} trace(s)."
        )
    else:
        table = Table(title="Wrong tool selections", title_justify="left", box=None, padding=(0, 2))
        table.add_column("called", style="red")
        table.add_column("judge says", style="green")
        table.add_column("n", justify="right")
        table.add_column("example", style="dim")
        for row in report.selection:
            table.add_row(row.called, row.judge_says, str(row.count), row.example_trace_id or "")
        console.print(table)

    console.print(
        f"  [dim]agreed {report.selection_agreed}"
        f" · not applicable {report.selection_skipped}"
        f" · invalid answers {report.selection_invalid}[/dim]"
    )


def _render_violations(console: Console, report: ToolCallReport) -> None:
    if not report.violations:
        console.print(
            f"[green]No illegal calls.[/green] "
            f"{report.violations_clean} trace(s) checked against the registry."
        )
    else:
        table = Table(title="Registry violations", title_justify="left", box=None, padding=(0, 2))
        table.add_column("code", style="red")
        table.add_column("tools")
        table.add_column("n", justify="right")
        table.add_column("example", style="dim")
        for row in report.violations:
            table.add_row(row.code, ", ".join(row.tools) or "—", str(row.count), row.example or "")
        console.print(table)

    console.print(
        f"  [dim]clean {report.violations_clean} · no tool calls {report.violations_skipped}[/dim]"
    )


def _render_contradictions(console: Console, report: ToolCallReport) -> None:
    if not report.contradictions:
        console.print(
            f"[green]No reply contradicted its calls.[/green] "
            f"{report.contradictions_consistent} trace(s) checked."
        )
    else:
        table = Table(
            title="Reply contradicts the calls", title_justify="left", box=None, padding=(0, 2)
        )
        table.add_column("trace", style="dim")
        table.add_column("called")
        table.add_column("contradiction", style="red")
        for row in report.contradictions:
            table.add_row(row.trace_id, row.called, row.explanation)
        console.print(table)

    console.print(
        f"  [dim]consistent {report.contradictions_consistent}"
        f" · no reply {report.contradictions_skipped}"
        f" · invalid answers {report.contradictions_invalid}[/dim]"
    )


def _render_failures(console: Console, report: ToolCallReport) -> None:
    if not report.failures:
        console.print(
            f"[green]No call failed at runtime.[/green] "
            f"{report.failures_clean} trace(s) recorded an outcome."
        )
    else:
        table = Table(
            title="Calls the tool rejected", title_justify="left", box=None, padding=(0, 2)
        )
        table.add_column("tool", style="red")
        table.add_column("error")
        table.add_column("n", justify="right")
        table.add_column("example", style="dim")
        for row in report.failures:
            table.add_row(row.tool, row.error, str(row.count), row.example or "")
        console.print(table)

    # Coverage first, because on most datasets it is the headline: a product
    # that logs invocations and not returns cannot answer this at all.
    console.print(
        f"  [dim]succeeded {report.failures_clean}"
        f" · no outcome recorded {report.failures_skipped}[/dim]"
    )
