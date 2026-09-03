"""`evalloop evaluate`.

Discovers `project.yaml` and `judges.yaml` beside the suite file, because that
is how an example directory is already laid out and because naming three files
on every invocation is the kind of friction that gets a tool abandoned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table
from sqlalchemy import Engine, select

from evalloop.cli.validate import render_problems
from evalloop.config import ConfigKind, Problem, load_config
from evalloop.contracts.judgeconf import JudgeConfig
from evalloop.contracts.project import ProjectConfig
from evalloop.contracts.suite import EvalSuite
from evalloop.contracts.trace import Trace
from evalloop.evaluate.registry import build_suite
from evalloop.evaluate.runner import RunSummary, run_suite
from evalloop.judge.cache import PostgresCache
from evalloop.store.artifacts import LocalArtifactStore
from evalloop.store.db import make_engine, session_scope
from evalloop.store.models import Project, Snapshot, TraceRow
from evalloop.store.repo import record_results, start_run, upsert_judge_config
from evalloop.store.traces import read_traces

__all__ = ["evaluate_command"]

_SIBLINGS = {ConfigKind.JUDGES: "judges.yaml", ConfigKind.PROJECT: "project.yaml"}


@dataclass(frozen=True, slots=True)
class _Configs:
    suite: EvalSuite
    project: ProjectConfig
    judges: dict[str, JudgeConfig]


def evaluate_command(
    suite_file: Annotated[Path, typer.Argument(help="Path to eval-suite.yaml")],
    split: Annotated[str, typer.Option("--split", help="Which split to evaluate.")] = "train",
    snapshot_id: Annotated[
        str | None,
        typer.Option("--snapshot", help="Snapshot to evaluate. Defaults to the project's latest."),
    ] = None,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Evaluate at most this many traces.")
    ] = None,
    no_cache: Annotated[
        bool, typer.Option("--no-cache", help="Ignore cached judge answers.")
    ] = False,
) -> None:
    """Run a suite of checks against a snapshot."""
    console = Console()
    errors = Console(stderr=True)

    configs, problems = _load_configs(suite_file)
    if configs is None:
        render_problems(errors, [], problems)
        raise typer.Exit(1)

    engine = make_engine()
    built = build_suite(
        configs.suite,
        configs.judges,
        cache=None if no_cache else PostgresCache(engine),
    )
    if not built.ok:
        for message in built.errors:
            errors.print(f"[red]✗[/red] {message}")
        raise typer.Exit(1)

    try:
        traces, resolved_snapshot = _load_traces(
            engine,
            project_name=configs.project.name,
            snapshot_id=snapshot_id,
            split=split,
            limit=limit,
        )
    except LookupError as exc:
        errors.print(f"[red]✗[/red] {exc}")
        raise typer.Exit(1) from exc

    summary = run_suite(traces, built)

    with session_scope(engine) as session:
        # Registered before the results that reference them. eval_result has a
        # foreign key onto judge_config.hash, which is rule 2 enforced by the
        # schema: a result cannot exist without the judge configuration that
        # produced it being on record.
        for client in built.judges.values():
            upsert_judge_config(
                session,
                version_hash=client.version_hash,
                provider=client.config.provider,
                model=client.config.model,
                params=client.config.identity(),
                response_schema=client.response_schema,
                system_prompt=client.system_prompt,
                rubric={"questions": client.questions},
            )

        run = start_run(
            session,
            snapshot_id=resolved_snapshot,
            suite_hash=configs.suite.suite_hash(),
            split=split,
        )
        run_id = run.id
        record_results(session, run_id=run_id, results=summary.results)
        run.status = "completed"

    _render(console, summary, run_id=run_id, snapshot_id=resolved_snapshot, split=split)
    raise typer.Exit(0)


def _load_configs(suite_file: Path) -> tuple[_Configs | None, list[Problem]]:
    """Load the suite and the two files beside it, reporting every problem."""
    problems: list[Problem] = []

    suite_config, suite_problems = load_config(suite_file, kind=ConfigKind.SUITE)
    problems.extend(suite_problems)

    directory = suite_file.resolve().parent
    loaded: dict[ConfigKind, object] = {}
    for kind, filename in _SIBLINGS.items():
        path = directory / filename
        if not path.exists():
            problems.append(
                Problem(
                    file=str(path),
                    message=f"{filename} not found beside the suite file",
                    hint=f"evaluate needs {filename} in the same directory as the suite",
                )
            )
            continue
        config, config_problems = load_config(path, kind=kind)
        problems.extend(config_problems)
        if config is not None:
            loaded[kind] = config.model

    if problems or suite_config is None:
        return None, problems

    project = loaded[ConfigKind.PROJECT]
    judges = loaded[ConfigKind.JUDGES]
    assert isinstance(suite_config.model, EvalSuite)
    assert isinstance(project, ProjectConfig)
    assert isinstance(judges, dict)
    return _Configs(suite=suite_config.model, project=project, judges=judges), []


def _load_traces(
    engine: Engine,
    *,
    project_name: str,
    snapshot_id: str | None,
    split: str,
    limit: int | None,
) -> tuple[list[Trace], str]:
    """Read traces for a snapshot from the artifact store.

    Postgres holds the pointers and the split assignment; the bodies come from
    Parquet. Filtering happens against the pointer rows so a split is a query
    rather than a scan of every trace body.
    """
    store = LocalArtifactStore(os.environ.get("EVALLOOP_ARTIFACT_ROOT", "./artifacts"))

    with session_scope(engine) as session:
        if snapshot_id is None:
            project = session.scalar(select(Project).where(Project.name == project_name))
            if project is None:
                raise LookupError(
                    f"project '{project_name}' has not been ingested yet; run evalloop ingest first"
                )
            snapshot = session.scalar(
                select(Snapshot)
                .where(Snapshot.project_id == project.id)
                .order_by(Snapshot.created_at.desc())
                .limit(1)
            )
            if snapshot is None:
                raise LookupError(f"project '{project_name}' has no snapshots yet")
            snapshot_id = snapshot.id

        rows = session.execute(
            select(TraceRow.trace_id, TraceRow.parquet_path)
            .where(TraceRow.snapshot_id == snapshot_id, TraceRow.split == split)
            .order_by(TraceRow.trace_id)
        ).all()

    if not rows:
        raise LookupError(f"snapshot {snapshot_id} has no traces in split '{split}'")

    uri = rows[0][1]
    if uri is None:
        raise LookupError(f"snapshot {snapshot_id} has no stored trace bodies")

    wanted = {trace_id for trace_id, _ in rows}
    traces = [t for t in read_traces(store, uri) if t.trace_id in wanted]
    return traces[:limit] if limit else traces, snapshot_id


def _render(
    console: Console,
    summary: RunSummary,
    *,
    run_id: str,
    snapshot_id: str,
    split: str,
) -> None:
    table = Table(title=None, header_style="bold")
    table.add_column("question")
    table.add_column("pass", justify="right")
    table.add_column("fail", justify="right")
    table.add_column("n/a", justify="right")
    table.add_column("err", justify="right")
    table.add_column("rate", justify="right")
    table.add_column("cost", justify="right")

    for question in summary.questions.values():
        rate = question.pass_rate
        table.add_row(
            question.evaluator_id,
            str(question.passed),
            f"[red]{question.failed}[/red]" if question.failed else "0",
            # Dimmed because it is neither good nor bad news - it is coverage,
            # and it belongs next to the numbers it qualifies.
            f"[yellow]{question.not_applicable}[/yellow]" if question.not_applicable else "0",
            f"[red]{question.errored}[/red]" if question.errored else "0",
            "—" if rate is None else f"{rate:.0%}",
            "—" if question.cost_usd == 0 else f"${question.cost_usd:.4f}",
        )

    console.print(table)
    console.print(
        f"[dim]run[/dim] {run_id}  "
        f"[dim]snapshot[/dim] {snapshot_id}  "
        f"[dim]split[/dim] {split}  "
        f"[dim]traces[/dim] {summary.traces}  "
        f"[dim]cache hits[/dim] {summary.cache_hits}"
    )

    unapplied = [q for q in summary.questions.values() if q.not_applicable]
    if unapplied:
        console.print(
            "\n[yellow]![/yellow] "
            f"{sum(q.not_applicable for q in unapplied)} result(s) had nothing to compare "
            "against — no ground truth at the configured path.\n"
            "  [dim]Not counted as failures. This number is how much of the dataset "
            "these checks cannot see.[/dim]"
        )
