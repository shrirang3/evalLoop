"""P0 acceptance.

The criterion, from plan/000:

    `evalloop ingest examples/support-bot/project.yaml && evalloop evaluate
    examples/support-bot/eval-suite.yaml` runs a JSONL dataset through one exact
    matcher and one custom LLM question, writes rows to Postgres, and
    `evalloop validate` rejects a suite with a typo'd key pointing at the right
    line.

Each clause gets a test, and `test_the_p0_command_chain` runs the whole thing as
a user would. This is the file to point at when asking whether P0 is done, and
it deliberately drives the real CLI over the real shipped example rather than
calling internals - the thing being tested is the product, not the modules.

Only the judge is substituted, for `provider: mock`, so CI needs no API key and
the answers are the same every time.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from evalloop.cli.main import app
from evalloop.store.models import (
    EvalResultRow,
    EvalRun,
    JudgeConfigRow,
    Project,
    Snapshot,
    SplitAssignment,
    TraceRow,
)

pytestmark = pytest.mark.integration

runner = CliRunner()
EXAMPLE = Path("examples/support-bot")

TRACES = 14
EVALUATORS = 6

# Expected outcomes, hand-checkable against examples/support-bot/traces.jsonl.
# The mock judge always answers True, so `policy_followed` fails on exactly the
# traces where the recorded human verdict is False - which is one column of the
# confusion matrix P3 is built on.
EXPECTED = {
    # Needs no ground truth: every trace that called something gets a verdict.
    # The four `n/a` are the traces that called nothing at all - legal, but
    # never looked at (plan/002 section 2).
    "tool_registry_check": {"pass": 10, "fail": 0, "n/a": 4},
    "tool_call_match": {"pass": 5, "fail": 5, "n/a": 4},
    "tool_name_match": {"pass": 5, "fail": 2, "n/a": 7},
    # Also needs no ground truth, and reaches every trace because "no tool" is
    # one of its answers. The mock judge picks the alphabetically first tool in
    # the catalogue, so this fails everywhere the agent did something else.
    "tool_selection": {"pass": 1, "fail": 13, "n/a": 0},
    "policy_followed": {"pass": 5, "fail": 7, "n/a": 2},
    "escalation_correct": {"pass": 0, "fail": 0, "n/a": TRACES},
}


@pytest.fixture
def project(
    tmp_path: Path, unique_name: str, pg_engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """The shipped example, renamed, with a mock judge."""
    for name in ("project.yaml", "judges.yaml", "eval-suite.yaml", "tools.yaml", "traces.jsonl"):
        shutil.copy(EXAMPLE / name, tmp_path / name)

    for name, old, new in (
        ("project.yaml", "name: support-bot", f"name: {unique_name}"),
        ("judges.yaml", "provider: anthropic", "provider: mock"),
    ):
        path = tmp_path / name
        path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")

    monkeypatch.setenv("EVALLOOP_DATABASE_URL", pg_engine.url.render_as_string(False))
    monkeypatch.setenv("EVALLOOP_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    return tmp_path


@pytest.fixture
def session(pg_engine: Engine) -> Session:
    return sessionmaker(bind=pg_engine, expire_on_commit=False)()


def _flat(result: object) -> str:
    return " ".join(result.output.split())  # type: ignore[attr-defined]


# --- the whole chain -----------------------------------------------------


def test_the_p0_command_chain(project: Path, session: Session, unique_name: str) -> None:
    """validate -> ingest -> snapshot show -> evaluate, exactly as documented."""
    configs = sorted(str(p) for p in project.glob("*.yaml"))
    assert runner.invoke(app, ["validate", *configs]).exit_code == 0

    ingested = runner.invoke(app, ["ingest", str(project / "project.yaml")])
    assert ingested.exit_code == 0, ingested.output
    snapshot_id = next(w for w in ingested.output.split() if w.startswith("snap-"))

    shown = runner.invoke(app, ["snapshot", "show", snapshot_id])
    assert shown.exit_code == 0
    assert f"split:train {TRACES}" in _flat(shown)

    evaluated = runner.invoke(
        app, ["evaluate", str(project / "eval-suite.yaml"), "--split", "train"]
    )
    assert evaluated.exit_code == 0, evaluated.output

    # Postgres now holds the answer to "how did this model do", by question.
    run_id = session.scalars(select(EvalRun.id).order_by(EvalRun.started_at.desc()).limit(1)).one()
    rows = session.execute(
        select(EvalResultRow.evaluator_id, EvalResultRow.passed, func.count())
        .where(EvalResultRow.run_id == run_id)
        .group_by(EvalResultRow.evaluator_id, EvalResultRow.passed)
    ).all()

    tally: dict[str, dict[str, int]] = {}
    for evaluator_id, passed, count in rows:
        key = {True: "pass", False: "fail", None: "n/a"}[passed]
        tally.setdefault(evaluator_id, {})[key] = count

    for evaluator_id, expected in EXPECTED.items():
        actual = {k: v for k, v in tally[evaluator_id].items() if v}
        assert actual == {k: v for k, v in expected.items() if v}, evaluator_id


# --- clause by clause ----------------------------------------------------


def test_it_runs_a_jsonl_dataset(project: Path, session: Session) -> None:
    """ "a JSONL dataset" - read through the mapping, not in trace shape already."""
    assert runner.invoke(app, ["ingest", str(project / "project.yaml")]).exit_code == 0

    snapshot = session.scalars(select(Snapshot).order_by(Snapshot.created_at.desc()).limit(1)).one()
    assert snapshot.row_count == TRACES

    trace = session.scalars(
        select(TraceRow).where(TraceRow.snapshot_id == snapshot.id).limit(1)
    ).first()
    assert trace is not None
    assert trace.parquet_path is not None and trace.parquet_path.startswith("cas://")


def test_it_runs_one_exact_matcher_and_one_llm_question(project: Path, session: Session) -> None:
    """ "one exact matcher and one custom LLM question" - both types, and the
    exact matcher has no judge while the question does."""
    assert runner.invoke(app, ["ingest", str(project / "project.yaml")]).exit_code == 0
    assert runner.invoke(app, ["evaluate", str(project / "eval-suite.yaml")]).exit_code == 0

    run_id = session.scalars(select(EvalRun.id).order_by(EvalRun.started_at.desc()).limit(1)).one()
    hashes = dict(
        session.execute(
            select(EvalResultRow.evaluator_id, func.min(EvalResultRow.judge_config_hash))
            .where(EvalResultRow.run_id == run_id)
            .group_by(EvalResultRow.evaluator_id)
        ).all()
    )

    assert hashes["tool_name_match"] is None  # deterministic: no judge at all
    assert hashes["policy_followed"] is not None


def test_it_writes_rows_to_postgres(project: Path, session: Session) -> None:
    """ "writes rows to Postgres" - every table the chain should touch."""
    assert runner.invoke(app, ["ingest", str(project / "project.yaml")]).exit_code == 0
    assert runner.invoke(app, ["evaluate", str(project / "eval-suite.yaml")]).exit_code == 0

    run_id = session.scalars(select(EvalRun.id).order_by(EvalRun.started_at.desc()).limit(1)).one()
    snapshot_id = session.scalars(
        select(Snapshot.id).order_by(Snapshot.created_at.desc()).limit(1)
    ).one()

    def count(model: object, *where: object) -> int | None:
        return session.scalar(select(func.count()).select_from(model).where(*where))  # type: ignore[arg-type]

    assert count(Project, Project.name.is_not(None)) >= 1  # type: ignore[operator]
    assert count(TraceRow, TraceRow.snapshot_id == snapshot_id) == TRACES
    assert count(SplitAssignment, SplitAssignment.snapshot_id == snapshot_id) == TRACES
    assert count(EvalResultRow, EvalResultRow.run_id == run_id) == TRACES * EVALUATORS
    assert count(JudgeConfigRow, JudgeConfigRow.hash.is_not(None)) >= 2  # type: ignore[operator]


def test_validate_rejects_a_typod_key_at_the_right_line(project: Path) -> None:
    """ "rejects a suite with a typo'd key pointing at the right line".

    The reason unknown keys are errors rather than warnings: `expcted` would
    otherwise be dropped in silence, the check would compare against nothing,
    and the suite would pass everything while looking healthy.
    """
    suite_path = project / "eval-suite.yaml"
    original = suite_path.read_text(encoding="utf-8")
    suite_path.write_text(
        original.replace(
            "    expected: ground_truth.tool_calls", "    expcted: ground_truth.tool_calls"
        ),
        encoding="utf-8",
    )
    expected_line = next(
        i
        for i, line in enumerate(suite_path.read_text(encoding="utf-8").splitlines(), 1)
        if "expcted:" in line
    )

    result = runner.invoke(app, ["validate", str(suite_path)])
    output = _flat(result)

    assert result.exit_code == 1
    assert "unknown key" in output
    assert f":{expected_line}" in output
    assert "did you mean 'expected'?" in output


# --- properties the phase claims -----------------------------------------


def test_re_ingesting_writes_nothing(project: Path, session: Session) -> None:
    """Otherwise a repeated ingest silently doubles a dataset and halves every
    rate computed from it, with nothing to notice."""
    first = runner.invoke(app, ["ingest", str(project / "project.yaml")])
    second = runner.invoke(app, ["ingest", str(project / "project.yaml")])

    assert "created" in _flat(first)
    assert "already ingested" in _flat(second)
    assert session.scalar(select(func.count()).select_from(Snapshot)) is not None
    assert next(w for w in first.output.split() if w.startswith("snap-")) == next(
        w for w in second.output.split() if w.startswith("snap-")
    )


def test_every_result_records_the_version_of_the_check_that_produced_it(
    project: Path, session: Session
) -> None:
    """Rule 3. Without it, "the metric moved" and "the check changed underneath
    me" are the same observation."""
    assert runner.invoke(app, ["ingest", str(project / "project.yaml")]).exit_code == 0
    assert runner.invoke(app, ["evaluate", str(project / "eval-suite.yaml")]).exit_code == 0

    missing = session.scalar(
        select(func.count())
        .select_from(EvalResultRow)
        .where(EvalResultRow.evaluator_version.is_(None))
    )
    assert missing == 0


def test_traces_with_no_ground_truth_are_not_recorded_as_failures(
    project: Path, session: Session
) -> None:
    """The decision the whole design turns on. `false` here would be a verdict
    the data does not support, and P4 would compile it into training data."""
    assert runner.invoke(app, ["ingest", str(project / "project.yaml")]).exit_code == 0
    assert runner.invoke(app, ["evaluate", str(project / "eval-suite.yaml")]).exit_code == 0

    run_id = session.scalars(select(EvalRun.id).order_by(EvalRun.started_at.desc()).limit(1)).one()
    unapplied = session.scalar(
        select(func.count())
        .select_from(EvalResultRow)
        .where(EvalResultRow.run_id == run_id, EvalResultRow.passed.is_(None))
    )
    # 4 + 7 + 2 + 14 from EXPECTED, all reported as coverage rather than failure.
    assert unapplied == sum(e["n/a"] for e in EXPECTED.values())


def test_the_shipped_example_is_the_one_under_test(project: Path) -> None:
    """The example is documentation people copy, so it is checked like code.
    Only judges.yaml and the project name differ from what ships."""
    for name in ("eval-suite.yaml", "tools.yaml", "traces.jsonl"):
        assert (project / name).read_bytes() == (EXAMPLE / name).read_bytes()
