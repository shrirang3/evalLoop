"""End-to-end ingest against real Postgres.

Snapshot idempotency is the property under test. Without it a repeated
`evalloop ingest` silently doubles a dataset and halves every rate computed
from it, with nothing to notice.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from evalloop.cli.main import app
from evalloop.contracts import ProjectConfig
from evalloop.ingest.pipeline import ingest
from evalloop.store import LocalArtifactStore, read_traces
from evalloop.store.models import Snapshot, SplitAssignment, TraceRow

pytestmark = pytest.mark.integration

ROW = {
    "id": "sb-0417",
    "user_transcript": "I want a refund on a 45 day old order",
    "assistant_transcript": "Sure, refunded.",
    "tool_calls": [{"name": "issue_refund", "arguments": {"order_id": "ORD-1"}}],
    "expected_reply": "Our refund window is 30 days.",
    "human_policy_verdict": False,
    "language": "en",
}

MAPPING = {
    "trace_id": "id",
    "input.user_request": "user_transcript",
    "output.text": "assistant_transcript",
    "output.tool_calls": "tool_calls",
    "ground_truth.expected_response": "expected_reply",
    "ground_truth.policy_followed": "human_policy_verdict",
    "metadata.language": "language",
}


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    rows = [ROW, {**ROW, "id": "sb-0418", "language": "hi"}, {**ROW, "id": "sb-0419"}]
    with (tmp_path / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return tmp_path


def _project(name: str = "ingest-test") -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "name": name,
            "source": {"type": "jsonl", "path": "traces.jsonl"},
            "mapping": MAPPING,
        }
    )


def _run(
    engine: Engine,
    project_dir: Path,
    *,
    name: str = "ingest-test",
    **kwargs: object,
) -> object:
    return ingest(
        _project(name),
        engine=engine,
        artifact_store=LocalArtifactStore(project_dir / "artifacts"),
        root=project_dir,
        config_yaml="name: ingest-test",
        **kwargs,  # type: ignore[arg-type]
    )


@pytest.fixture
def isolated(pg_engine: Engine, project_dir: Path) -> Session:
    """A session for assertions. Ingest opens its own, so rows really commit -
    which is why each test uses a uniquely named project and cleans up after."""
    factory = sessionmaker(bind=pg_engine, expire_on_commit=False)
    return factory()


def test_ingest_writes_traces_splits_and_a_snapshot(
    pg_engine: Engine, project_dir: Path, isolated: Session, unique_name: str
) -> None:
    report = _run(pg_engine, project_dir, name=unique_name)

    assert report.created is True  # type: ignore[attr-defined]
    assert report.row_count == 3  # type: ignore[attr-defined]
    snapshot_id = report.snapshot_id  # type: ignore[attr-defined]

    snapshot = isolated.get(Snapshot, snapshot_id)
    assert snapshot is not None and snapshot.row_count == 3

    traces = isolated.scalars(select(TraceRow).where(TraceRow.snapshot_id == snapshot_id)).all()
    assert {t.trace_id for t in traces} == {"sb-0417", "sb-0418", "sb-0419"}
    assert all(t.parquet_path is not None for t in traces)
    assert all(t.split == "train" for t in traces)

    assignments = isolated.scalars(
        select(SplitAssignment).where(SplitAssignment.snapshot_id == snapshot_id)
    ).all()
    assert len(assignments) == 3
    assert not any(a.sealed for a in assignments)


def test_reingesting_identical_data_writes_nothing(
    pg_engine: Engine, project_dir: Path, isolated: Session, unique_name: str
) -> None:
    first = _run(pg_engine, project_dir, name=unique_name)
    second = _run(pg_engine, project_dir, name=unique_name)

    assert first.snapshot_id == second.snapshot_id  # type: ignore[attr-defined]
    assert second.created is False  # type: ignore[attr-defined]
    assert second.row_count == 3  # type: ignore[attr-defined]

    total = isolated.scalar(
        select(func.count()).select_from(TraceRow).where(TraceRow.snapshot_id == first.snapshot_id)  # type: ignore[attr-defined]
    )
    assert total == 3


def test_reingest_does_not_leave_an_orphan_artifact(
    pg_engine: Engine, project_dir: Path, unique_name: str
) -> None:
    """The snapshot is looked up before the Parquet is written. Producing a
    multi-gigabyte file and then finding the snapshot already exists would
    leave an artifact nothing references."""
    first = _run(pg_engine, project_dir, name=unique_name)
    second = _run(pg_engine, project_dir, name=unique_name)

    assert first.traces_uri == second.traces_uri  # type: ignore[attr-defined]
    files = [p for p in (project_dir / "artifacts").rglob("*") if p.is_file()]
    assert len(files) == 1


def test_changed_source_data_creates_a_second_snapshot(
    pg_engine: Engine, project_dir: Path, unique_name: str
) -> None:
    first = _run(pg_engine, project_dir, name=unique_name)

    with (project_dir / "traces.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**ROW, "id": "sb-0420"}) + "\n")

    second = _run(pg_engine, project_dir, name=unique_name)
    assert second.snapshot_id != first.snapshot_id  # type: ignore[attr-defined]
    assert second.created is True  # type: ignore[attr-defined]
    assert second.row_count == 4  # type: ignore[attr-defined]


def test_traces_round_trip_through_the_artifact_store(
    pg_engine: Engine, project_dir: Path, unique_name: str
) -> None:
    report = _run(pg_engine, project_dir, name=unique_name)
    store = LocalArtifactStore(project_dir / "artifacts")

    traces = read_traces(store, report.traces_uri)  # type: ignore[attr-defined]
    assert len(traces) == 3

    trace = next(t for t in traces if t.trace_id == "sb-0417")
    assert trace.output.tool_calls[0].name == "issue_refund"
    assert trace.ground_truth.get("policy_followed") is False
    assert trace.metadata["language"] == "en"
    assert trace.source_id == "line:1"


def test_dry_run_writes_nothing(
    pg_engine: Engine, project_dir: Path, isolated: Session, unique_name: str
) -> None:
    before = isolated.scalar(select(func.count()).select_from(Snapshot))
    report = _run(pg_engine, project_dir, name=unique_name, dry_run=True)

    assert report.row_count == 3  # type: ignore[attr-defined]
    assert report.snapshot_id is None  # type: ignore[attr-defined]
    assert report.fingerprint  # computed, so a dry run still tells you the identity
    isolated.expire_all()
    assert isolated.scalar(select(func.count()).select_from(Snapshot)) == before
    assert not (project_dir / "artifacts").exists()


def test_empty_source_is_an_error_not_a_silent_empty_snapshot(
    pg_engine: Engine, tmp_path: Path, unique_name: str
) -> None:
    """An empty snapshot would evaluate to a perfect score on zero traces."""
    (tmp_path / "traces.jsonl").write_text("", encoding="utf-8")
    report = _run(pg_engine, tmp_path, name=unique_name)
    assert report.snapshot_id is None  # type: ignore[attr-defined]
    assert "no traces" in report.errors[0]  # type: ignore[attr-defined]


# --- CLI, against the real metastore ---

runner = CliRunner()


def test_snapshot_show_rejects_an_unknown_id(
    pg_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALLOOP_DATABASE_URL", str(pg_engine.url.render_as_string(False)))
    result = runner.invoke(app, ["snapshot", "show", "snap-does-not-exist"])
    assert result.exit_code == 1
    assert "no such snapshot" in result.output


def test_ingest_then_snapshot_show(
    pg_engine: Engine,
    project_dir: Path,
    unique_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two commands a P0 user actually chains together."""
    monkeypatch.setenv("EVALLOOP_DATABASE_URL", str(pg_engine.url.render_as_string(False)))
    monkeypatch.setenv("EVALLOOP_ARTIFACT_ROOT", str(project_dir / "artifacts"))

    (project_dir / "project.yaml").write_text(
        f"""name: {unique_name}
source:
  type: jsonl
  path: traces.jsonl
mapping:
  trace_id: id
  input.user_request: user_transcript
  output.text: assistant_transcript
  output.tool_calls: tool_calls
  ground_truth.expected_response: expected_reply
  ground_truth.policy_followed: human_policy_verdict
  metadata.language: language
""",
        encoding="utf-8",
    )

    ingested = runner.invoke(app, ["ingest", str(project_dir / "project.yaml")])
    assert ingested.exit_code == 0, ingested.output
    assert "created" in ingested.output

    snapshot_id = next(word for word in ingested.output.split() if word.startswith("snap-"))
    shown = runner.invoke(app, ["snapshot", "show", snapshot_id])
    assert shown.exit_code == 0
    output = " ".join(shown.output.split())
    assert "split:train 3" in output
    assert "cas://" in output


def test_second_ingest_via_cli_reports_nothing_written(
    pg_engine: Engine,
    project_dir: Path,
    unique_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVALLOOP_DATABASE_URL", str(pg_engine.url.render_as_string(False)))
    monkeypatch.setenv("EVALLOOP_ARTIFACT_ROOT", str(project_dir / "artifacts"))
    (project_dir / "project.yaml").write_text(
        f"name: {unique_name}\nsource:\n  type: jsonl\n  path: traces.jsonl\n"
        "mapping:\n  trace_id: id\n  input.user_request: user_transcript\n",
        encoding="utf-8",
    )

    assert runner.invoke(app, ["ingest", str(project_dir / "project.yaml")]).exit_code == 0
    again = runner.invoke(app, ["ingest", str(project_dir / "project.yaml")])
    assert again.exit_code == 0
    assert "already ingested" in " ".join(again.output.split())
