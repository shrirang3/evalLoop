"""The ingest run: source rows in, an immutable snapshot out.

Order matters here. Traces are built and hashed *before* anything is written,
because the snapshot fingerprint is derived from their content hashes - so a
re-ingest of unchanged data is recognised and nothing is written at all. Writing
first and deduplicating afterwards would make idempotency a cleanup problem
rather than a property.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import Engine, select

from evalloop.contracts.project import ProjectConfig
from evalloop.contracts.trace import Trace
from evalloop.ingest.connectors.jsonl import JsonlConnector
from evalloop.ingest.mapping import apply_mapping
from evalloop.store.artifacts import LocalArtifactStore
from evalloop.store.db import session_scope
from evalloop.store.models import TraceRow
from evalloop.store.repo import (
    find_snapshot,
    source_fingerprint,
    upsert_project,
    upsert_snapshot,
)
from evalloop.store.traces import write_traces

__all__ = ["IngestReport", "build_traces", "ingest"]

_SUPPORTED = {"jsonl"}


@dataclass
class IngestReport:
    """What one ingest did, or would have done under `--dry-run`."""

    project: str
    snapshot_id: str | None = None
    created: bool = False
    """False when an identical snapshot already existed. Nothing was written."""

    row_count: int = 0
    skipped: int = 0
    fingerprint: str = ""
    traces_uri: str | None = None
    split: str = "train"

    errors: list[str] = field(default_factory=list)
    unmapped_fields: list[str] = field(default_factory=list)
    sample: list[Trace] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def build_traces(
    project: ProjectConfig,
    *,
    root: Path,
    limit: int | None = None,
) -> tuple[list[Trace], IngestReport]:
    """Read the source and map it, without touching the database.

    Split out so `--dry-run` runs exactly the code a real ingest runs, rather
    than a parallel path that can drift from it.
    """
    report = IngestReport(project=project.name)

    if project.source.type not in _SUPPORTED:
        report.errors.append(
            f"source type '{project.source.type}' is not implemented yet "
            f"(P0 supports: {', '.join(sorted(_SUPPORTED))})"
        )
        return [], report

    assert project.source.path is not None  # guaranteed by SourceConfig validation
    path = Path(project.source.path)
    if not path.is_absolute():
        path = root / path

    connector = JsonlConnector(path=path)
    rows = [(row.data, row.source_id) for row in connector.rows(limit=limit)]
    report.errors.extend(connector.errors)

    mapped = apply_mapping(rows, project.mapping)
    report.errors.extend(mapped.errors)
    report.unmapped_fields = sorted(mapped.unmapped_fields)
    report.skipped = mapped.skipped
    report.row_count = len(mapped.traces)
    report.sample = mapped.traces[:5]

    report.fingerprint = source_fingerprint(
        connector_config=connector.fingerprint_config(),
        content_hashes=[t.content_hash for t in mapped.traces],
    )
    return mapped.traces, report


def ingest(
    project: ProjectConfig,
    *,
    engine: Engine,
    artifact_store: LocalArtifactStore,
    root: Path,
    config_yaml: str,
    limit: int | None = None,
    dry_run: bool = False,
) -> IngestReport:
    """Ingest a project's source into an immutable snapshot."""
    traces, report = build_traces(project, root=root, limit=limit)

    if not traces:
        if not report.errors:
            report.errors.append("no traces were produced from the source")
        return report

    if dry_run:
        return report

    with session_scope(engine) as session:
        # Checked before writing anything. Producing a multi-gigabyte Parquet
        # file and then discovering the snapshot already exists would leave an
        # artifact nothing references.
        existing = find_snapshot(session, report.fingerprint)
        if existing is not None:
            report.snapshot_id = existing.id
            report.created = False
            report.row_count = existing.row_count
            report.traces_uri = session.scalar(
                select(TraceRow.parquet_path).where(TraceRow.snapshot_id == existing.id)
            )
            return report

        # Written before the snapshot row so a snapshot never points at an
        # artifact that does not exist. Content addressing makes an orphan from
        # a failed ingest harmless - the next attempt produces the same address
        # and reuses it.
        report.traces_uri = write_traces(artifact_store, traces, split=report.split)

        project_row = upsert_project(session, name=project.name, config_yaml=config_yaml)
        snapshot, created = upsert_snapshot(
            session,
            project_id=project_row.id,
            fingerprint=report.fingerprint,
            traces=traces,
            default_split=report.split,
        )
        report.snapshot_id = snapshot.id
        report.created = created

        session.query(TraceRow).filter(TraceRow.snapshot_id == snapshot.id).update(
            {TraceRow.parquet_path: report.traces_uri}
        )

    return report
