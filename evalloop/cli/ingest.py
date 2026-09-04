"""`evalloop ingest` and `evalloop snapshot show`."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import select

from evalloop.cli.validate import render_problems
from evalloop.config import ConfigKind, load_config
from evalloop.ingest.pipeline import IngestReport, ingest
from evalloop.store.artifacts import LocalArtifactStore
from evalloop.store.db import make_engine, session_scope
from evalloop.store.models import Snapshot, TraceRow

__all__ = ["ingest_command", "snapshot_app"]

_DEFAULT_ARTIFACT_DIR = "./artifacts"


def _artifact_store() -> LocalArtifactStore:
    return LocalArtifactStore(os.environ.get("EVALLOOP_ARTIFACT_ROOT", _DEFAULT_ARTIFACT_DIR))


def ingest_command(
    project_file: Annotated[Path, typer.Argument(help="Path to project.yaml")],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Map and report without writing anything."),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option("--limit", help="Read at most this many source rows."),
    ] = None,
) -> None:
    """Read a project's traces into an immutable snapshot."""
    console = Console()
    problems_console = Console(stderr=True)

    config, problems = load_config(project_file, kind=ConfigKind.PROJECT)
    if config is None:
        render_problems(problems_console, [], problems)
        raise typer.Exit(1)

    report = ingest(
        config.model,
        # Not built for a dry run: it needs no database, so it must not require
        # one to be reachable or even configured.
        engine=None if dry_run else make_engine(),
        artifact_store=_artifact_store(),
        # Source paths are resolved relative to project.yaml, so a project
        # directory can be moved or checked out anywhere and still work.
        root=project_file.resolve().parent,
        config_yaml=config.source.text,
        limit=limit,
        dry_run=dry_run,
    )

    _render_report(console, report, dry_run=dry_run)
    raise typer.Exit(0 if report.ok else 1)


def _render_report(console: Console, report: IngestReport, *, dry_run: bool) -> None:
    if dry_run:
        console.print("[yellow]dry run[/yellow] — nothing written\n")

    if report.errors:
        console.print(f"[red]{len(report.errors)} problem(s):[/red]")
        for error in report.errors[:20]:
            console.print(f"  [red]✗[/red] {error}")
        if len(report.errors) > 20:
            console.print(f"  [dim]… and {len(report.errors) - 20} more[/dim]")
        console.print()

    if report.unmapped_fields:
        # Surfaced because a forgotten mapping line is otherwise invisible until
        # someone wonders why a slice came back empty.
        console.print(
            f"[yellow]![/yellow] source fields with no mapping: "
            f"[dim]{', '.join(report.unmapped_fields)}[/dim]\n"
        )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("project", report.project)
    table.add_row("traces", str(report.row_count))
    if report.skipped:
        table.add_row("skipped", f"[red]{report.skipped}[/red]")
    table.add_row("fingerprint", f"[dim]{report.fingerprint[:16]}…[/dim]")

    if report.snapshot_id:
        table.add_row("snapshot", report.snapshot_id)
        table.add_row(
            "status",
            "[green]created[/green]"
            if report.created
            else "[yellow]already ingested — nothing written[/yellow]",
        )
    if report.traces_uri:
        table.add_row("traces", f"[dim]{report.traces_uri}[/dim]")
    console.print(table)

    if dry_run and report.sample:
        console.print("\n[bold]first mapped trace[/bold]")
        console.print_json(report.sample[0].model_dump_json(indent=2))


snapshot_app = typer.Typer(help="Inspect snapshots.", no_args_is_help=True)


@snapshot_app.command("show")
def snapshot_show(
    snapshot_id: Annotated[str, typer.Argument(help="Snapshot id")],
) -> None:
    """Split sizes and provenance for one snapshot."""
    console = Console()
    with session_scope(make_engine()) as session:
        snapshot = session.get(Snapshot, snapshot_id)
        if snapshot is None:
            console.print(f"[red]no such snapshot:[/red] {snapshot_id}")
            raise typer.Exit(1)

        splits = session.execute(
            select(TraceRow.split, TraceRow.parquet_path).where(TraceRow.snapshot_id == snapshot_id)
        ).all()

        counts: dict[str, int] = {}
        for split, _ in splits:
            counts[split] = counts.get(split, 0) + 1

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_row("id", snapshot.id)
        table.add_row("project", snapshot.project_id)
        table.add_row("rows", str(snapshot.row_count))
        table.add_row("fingerprint", f"[dim]{snapshot.source_fingerprint[:16]}…[/dim]")
        table.add_row("created", str(snapshot.created_at))
        for split, count in sorted(counts.items()):
            table.add_row(f"split:{split}", str(count))
        if splits and splits[0][1]:
            table.add_row("traces", f"[dim]{splits[0][1]}[/dim]")
        console.print(table)
