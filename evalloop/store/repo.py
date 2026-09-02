"""Repository functions over the metastore.

Thin on purpose: SQLAlchemy is not hidden behind an abstraction, only the
operations that carry a rule worth enforcing in one place. Chief among them is
snapshot idempotency, which is what stops a repeated `evalloop ingest` from
silently doubling a dataset and quietly halving every rate computed from it.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from evalloop.contracts.trace import Trace, canonical_json
from evalloop.store.models import (
    EvalResultRow,
    EvalRun,
    JudgeConfigRow,
    Project,
    Snapshot,
    SplitAssignment,
    TraceRow,
)

__all__ = [
    "find_snapshot",
    "new_id",
    "record_results",
    "source_fingerprint",
    "start_run",
    "upsert_judge_config",
    "upsert_project",
    "upsert_snapshot",
]


def new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex[:16]
    return f"{prefix}-{raw}" if prefix else raw


def source_fingerprint(
    *,
    connector_config: dict[str, Any],
    content_hashes: Iterable[str],
) -> str:
    """Fingerprint identifying "this exact data from this exact source".

    Content hashes are sorted before hashing, so row order from the source
    cannot change the fingerprint. A query returning the same rows in a
    different order is the same snapshot.
    """
    hashes = sorted(content_hashes)
    payload = {
        "connector": connector_config,
        "row_count": len(hashes),
        "content_hashes": hashes,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def find_snapshot(session: Session, fingerprint: str) -> Snapshot | None:
    """The snapshot for this exact source data, if it was already ingested.

    Exposed separately from `upsert_snapshot` so a caller can find out *before*
    doing expensive work - writing a multi-gigabyte Parquet file and then
    discovering the snapshot already exists leaves an artifact nothing
    references.
    """
    return session.scalar(select(Snapshot).where(Snapshot.source_fingerprint == fingerprint))


def upsert_project(session: Session, *, name: str, config_yaml: str) -> Project:
    """Fetch a project by name, or create it.

    `config_yaml` is refreshed on an existing project: the config evolves, and a
    run records the snapshot and suite hashes it actually used, so the project
    row does not need to be frozen.
    """
    existing = session.scalar(select(Project).where(Project.name == name))
    if existing is not None:
        existing.config_yaml = config_yaml
        return existing

    project = Project(id=new_id("proj"), name=name, config_yaml=config_yaml)
    session.add(project)
    session.flush()
    return project


def upsert_snapshot(
    session: Session,
    *,
    project_id: str,
    fingerprint: str,
    traces: Sequence[Trace],
    default_split: str = "train",
    split_strategy: str = "none",
) -> tuple[Snapshot, bool]:
    """Create a snapshot, or return the existing one with the same fingerprint.

    Returns `(snapshot, created)`. When `created` is False nothing was written -
    the caller is looking at the snapshot a previous ingest produced.

    Splits are not assigned here beyond `default_split`; strategy-driven
    splitting is P1. The rows are written now so the unique constraint on
    (snapshot_id, trace_id) is in force from the first ingest.
    """
    existing = find_snapshot(session, fingerprint)
    if existing is not None:
        return existing, False

    snapshot = Snapshot(
        id=new_id("snap"),
        project_id=project_id,
        source_fingerprint=fingerprint,
        row_count=len(traces),
    )
    session.add(snapshot)
    session.flush()

    session.add_all(
        TraceRow(
            snapshot_id=snapshot.id,
            trace_id=trace.trace_id,
            split=default_split,
            content_hash=trace.content_hash,
        )
        for trace in traces
    )
    session.add_all(
        SplitAssignment(
            snapshot_id=snapshot.id,
            trace_id=trace.trace_id,
            split=default_split,
            split_strategy=split_strategy,
            sealed=default_split == "test",
        )
        for trace in traces
    )
    session.flush()
    return snapshot, True


def upsert_judge_config(
    session: Session,
    *,
    version_hash: str,
    provider: str,
    model: str,
    params: dict[str, Any],
    response_schema: dict[str, Any],
    system_prompt: str | None = None,
    rubric: dict[str, Any] | None = None,
) -> JudgeConfigRow:
    """Register a judge configuration under its version hash.

    Idempotent by construction: the hash is the primary key, so re-registering
    an identical judge is a no-op and a changed rubric is a different row rather
    than an overwrite of the old one.
    """
    existing = session.get(JudgeConfigRow, version_hash)
    if existing is not None:
        return existing

    row = JudgeConfigRow(
        hash=version_hash,
        provider=provider,
        model=model,
        params=params,
        system_prompt=system_prompt,
        rubric=rubric,
        response_schema=response_schema,
    )
    session.add(row)
    session.flush()
    return row


def start_run(
    session: Session,
    *,
    snapshot_id: str,
    suite_hash: str,
    split: str,
    model_name: str | None = None,
) -> EvalRun:
    run = EvalRun(
        id=new_id("run"),
        snapshot_id=snapshot_id,
        suite_hash=suite_hash,
        split=split,
        model_name=model_name,
    )
    session.add(run)
    session.flush()
    return run


def record_results(session: Session, *, run_id: str, results: Iterable[Any]) -> int:
    """Write EvalResult contracts as rows, and roll their cost up onto the run.

    Aggregates are maintained here rather than recomputed later so a partial run
    still reports what it spent. A run aborted by a budget limit has to be able
    to say how much it used.
    """
    run = session.get(EvalRun, run_id)
    if run is None:
        raise LookupError(f"no such run: {run_id}")

    count = 0
    for result in results:
        session.add(
            EvalResultRow(
                run_id=run_id,
                trace_id=result.trace_id,
                evaluator_id=result.evaluator_id,
                evaluator_version=result.evaluator_version,
                score=result.score,
                passed=result.passed,
                normalized_prediction=_jsonable(result.normalized_prediction),
                ground_truth=_jsonable(result.ground_truth),
                explanation=result.explanation,
                raw_output=result.raw_output,
                judge_config_hash=result.judge_config_hash,
                cache_hit=result.cache_hit,
                invalid_output=result.invalid_output,
                error=result.error,
                latency_ms=result.latency_ms,
                cost_usd=result.cost_usd,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
            )
        )
        run.cost_usd += result.cost_usd or 0.0
        run.tokens_in += result.tokens_in or 0
        run.tokens_out += result.tokens_out or 0
        run.cache_hits += 1 if result.cache_hit else 0
        count += 1

    session.flush()
    return count


def _jsonable(value: Any) -> dict[str, Any] | None:
    """Wrap a scalar prediction so it fits a JSON column.

    Predictions are arbitrary - a bool, a string, a list of tool calls. Storing
    them under a `value` key keeps the column typed as an object rather than
    forcing every reader to handle four shapes.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return {"value": value}
