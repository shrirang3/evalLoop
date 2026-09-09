"""`evalloop feedback build` - failures out, training rows in.

The report is where the number gets read. This is what the number is *for*: a
`tool_selection` failure already carries the correct call, so a preference pair
needs no labels and no human.

Reads the run's results and the snapshot's traces, because a training row needs
both - the verdict and the judge's proposal come from the result, the prompt and
the call that actually happened come from the trace.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import Engine, select

from evalloop.config import ConfigKind, load_config
from evalloop.contracts.result import EvalResult
from evalloop.contracts.tools import ToolRegistry
from evalloop.contracts.trace import Trace
from evalloop.feedback import Dataset, build_dpo
from evalloop.store.artifacts import LocalArtifactStore
from evalloop.store.db import make_engine, session_scope
from evalloop.store.models import EvalResultRow, EvalRun, TraceRow
from evalloop.store.traces import read_traces

__all__ = ["feedback_app"]

feedback_app = typer.Typer(help="Compile failures into training data.", no_args_is_help=True)

_DEFAULT_ARTIFACT_DIR = "./artifacts"
_SEALED_SPLIT = "test"


@feedback_app.command("build")
def feedback_build(
    run_id: Annotated[
        str | None,
        typer.Argument(help="Run to compile. Defaults to the most recent run."),
    ] = None,
    out: Annotated[
        Path,
        typer.Option("--out", help="Where to write the JSONL dataset."),
    ] = Path("feedback.jsonl"),
    strategy: Annotated[
        str,
        typer.Option("--strategy", help="Only 'dpo' is implemented."),
    ] = "dpo",
    selection_id: Annotated[
        str,
        typer.Option("--selection", help="Evaluator id of the tool_selection check."),
    ] = "tool_selection",
    tools_file: Annotated[
        Path | None,
        typer.Option("--tools", help="tools.yaml, used to validate the judge's proposals."),
    ] = None,
) -> None:
    """Compile a run's tool-selection failures into DPO preference pairs."""
    console = Console()
    errors = Console(stderr=True)

    if strategy != "dpo":
        errors.print(
            f"[red]strategy '{strategy}' is not implemented[/red] — "
            "sft arrives with a target source that produces full responses"
        )
        raise typer.Exit(1)

    registry = _registry(tools_file, errors)
    engine = make_engine()

    with session_scope(engine) as session:
        resolved = (
            run_id
            or session.scalars(
                select(EvalRun.id).order_by(EvalRun.started_at.desc()).limit(1)
            ).first()
        )
        if resolved is None:
            errors.print("[red]no runs recorded[/red] — run `evalloop evaluate` first")
            raise typer.Exit(1)

        run = session.get(EvalRun, resolved)
        if run is None:
            errors.print(f"[red]no such run:[/red] {resolved}")
            raise typer.Exit(1)

        traces = _traces(session, engine, run.snapshot_id)
        sealed = _sealed(session, run.snapshot_id)
        results = _results(session, resolved, selection_id)

    if not results:
        errors.print(
            f"[red]run {resolved} has no '{selection_id}' results[/red] — "
            "the only target source implemented needs a tool_selection check in the suite"
        )
        raise typer.Exit(1)

    by_id = {trace.trace_id: trace for trace in traces}
    pairs = [(by_id[r.trace_id], r) for r in results if r.trace_id in by_id]
    dataset = build_dpo(pairs, registry, sealed_trace_ids=sealed)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dataset.to_jsonl(), encoding="utf-8")
    _render(console, dataset, resolved, out)

    if dataset.size == 0:
        # Not an error - a run where nothing failed produces no rows, and that
        # is the desirable outcome. Exit code says "nothing to train on".
        raise typer.Exit(2)


def _registry(path: Path | None, errors: Console) -> ToolRegistry | None:
    if path is None:
        return None
    config, problems = load_config(path, kind=ConfigKind.TOOLS)
    if config is None or not isinstance(config.model, ToolRegistry):
        for problem in problems:
            errors.print(f"[red]✗[/red] {problem.message}")
        raise typer.Exit(1)
    return config.model


def _traces(session: Any, engine: Engine, snapshot_id: str) -> list[Trace]:
    store = LocalArtifactStore(os.environ.get("EVALLOOP_ARTIFACT_ROOT", _DEFAULT_ARTIFACT_DIR))
    uri = session.scalars(
        select(TraceRow.parquet_path)
        .where(TraceRow.snapshot_id == snapshot_id, TraceRow.parquet_path.is_not(None))
        .limit(1)
    ).first()
    return read_traces(store, uri) if uri else []


def _sealed(session: Any, snapshot_id: str) -> frozenset[str]:
    """Trace ids in the sealed split. Rule 12 is enforced by exclusion, not hope."""
    return frozenset(
        session.scalars(
            select(TraceRow.trace_id).where(
                TraceRow.snapshot_id == snapshot_id, TraceRow.split == _SEALED_SPLIT
            )
        ).all()
    )


def _results(session: Any, run_id: str, evaluator_id: str) -> list[EvalResult]:
    rows = session.execute(
        select(
            EvalResultRow.trace_id,
            EvalResultRow.evaluator_id,
            EvalResultRow.evaluator_version,
            EvalResultRow.passed,
            EvalResultRow.normalized_prediction,
            EvalResultRow.ground_truth,
            EvalResultRow.raw_output,
            EvalResultRow.invalid_output,
            EvalResultRow.error,
            EvalResultRow.judge_config_hash,
        ).where(EvalResultRow.run_id == run_id, EvalResultRow.evaluator_id == evaluator_id)
    ).all()
    return [
        EvalResult(
            trace_id=row.trace_id,
            evaluator_id=row.evaluator_id,
            evaluator_version=row.evaluator_version,
            passed=row.passed,
            normalized_prediction=row.normalized_prediction,
            ground_truth=row.ground_truth,
            raw_output=row.raw_output,
            invalid_output=row.invalid_output,
            error=row.error,
            judge_config_hash=row.judge_config_hash,
        )
        for row in rows
    ]


def _render(console: Console, dataset: Dataset, run_id: str, out: Path) -> None:
    manifest = dataset.manifest()
    console.print()
    console.print(f"[bold]{dataset.size} preference pair(s)[/bold] from run {run_id} → {out}")
    console.print(f"[dim]fingerprint {manifest['fingerprint'][:16]}…[/dim]")

    if dataset.dropped:
        console.print()
        table = Table(title="Dropped", title_justify="left", box=None, padding=(0, 2))
        table.add_column("reason")
        table.add_column("n", justify="right")
        for reason, count in sorted(dataset.dropped.items(), key=lambda kv: (-kv[1], kv[0])):
            table.add_row(reason, str(count))
        console.print(table)

    console.print()
    console.print(
        "[dim]Every row carries target_source, signal_provenance, judge_version and\n"
        "judge_health. These are judge-derived: relative claims only until the judge\n"
        "is calibrated.[/dim]"
    )
