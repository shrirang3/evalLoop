"""plan/002 acceptance: tool correctness on a dataset with no ground truth.

The claim under test is narrow and is the product's whole wedge. Point EvalLoop
at production traces - transcripts, tool calls, nothing else - plus `tools.yaml`,
and the tool checks return real verdicts rather than abstaining.

The contrast with `tool_call_match` in the same run is the point. It needs a
recorded target, so it reports `not applicable` on every row here, which is
what it would do on any real customer's data.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from evalloop.cli.main import app
from evalloop.store.models import EvalResultRow, EvalRun

pytestmark = pytest.mark.integration

runner = CliRunner()

EXAMPLE = Path("examples/support-bot")

GROUND_TRUTH_COLUMNS = ("expected_tool_calls", "expected_reply", "human_policy_verdict")
"""The columns no product emits. Stripped here to make that concrete."""


@pytest.fixture
def env(
    pg_engine: Engine, tmp_path: Path, unique_name: str, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """The shipped example with every ground-truth column removed."""
    for name in ("project.yaml", "judges.yaml", "eval-suite.yaml", "tools.yaml"):
        shutil.copy(EXAMPLE / name, tmp_path / name)

    rows = [
        json.loads(line)
        for line in (EXAMPLE / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    (tmp_path / "traces.jsonl").write_text(
        "\n".join(
            json.dumps({k: v for k, v in row.items() if k not in GROUND_TRUTH_COLUMNS})
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )

    project = (tmp_path / "project.yaml").read_text(encoding="utf-8")
    (tmp_path / "project.yaml").write_text(
        project.replace("name: support-bot", f"name: {unique_name}"), encoding="utf-8"
    )
    judges = (tmp_path / "judges.yaml").read_text(encoding="utf-8")
    (tmp_path / "judges.yaml").write_text(
        judges.replace("provider: anthropic", "provider: mock"), encoding="utf-8"
    )

    monkeypatch.setenv("EVALLOOP_DATABASE_URL", pg_engine.url.render_as_string(False))
    monkeypatch.setenv("EVALLOOP_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    return tmp_path


@pytest.fixture
def session(pg_engine: Engine) -> Session:
    return sessionmaker(bind=pg_engine, expire_on_commit=False)()


def _run(env: Path, *args: str) -> str:
    ingest = runner.invoke(app, ["ingest", str(env / "project.yaml")])
    assert ingest.exit_code == 0, ingest.output
    evaluate = runner.invoke(
        app, ["evaluate", str(env / "eval-suite.yaml"), "--split", "train", *args]
    )
    assert evaluate.exit_code == 0, evaluate.output
    return " ".join(evaluate.output.split())


def _verdicts(session: Session, evaluator_id: str) -> list[bool | None]:
    session.expire_all()
    run_id = session.scalars(select(EvalRun.id).order_by(EvalRun.started_at.desc()).limit(1)).one()
    return list(
        session.scalars(
            select(EvalResultRow.passed).where(
                EvalResultRow.run_id == run_id,
                EvalResultRow.evaluator_id == evaluator_id,
            )
        ).all()
    )


def test_a_dataset_with_no_ground_truth_still_produces_tool_verdicts(
    env: Path, session: Session
) -> None:
    _run(env)

    registry_check = _verdicts(session, "tool_registry_check")
    selection = _verdicts(session, "tool_selection")

    assert len(registry_check) == 14
    # Traces that called nothing are `not applicable` - legal, but never looked
    # at, and counting them as passes would inflate the rate.
    assert any(passed is not None for passed in registry_check)
    # Selection reaches every trace, because "no tool" is one of its answers.
    assert all(passed is not None for passed in selection)


def test_the_ground_truth_check_abstains_on_the_same_run(env: Path, session: Session) -> None:
    """Side by side, in one run: this is the gap plan/002 exists to close."""
    _run(env)
    assert _verdicts(session, "tool_call_match") == [None] * 14


def test_the_run_costs_money_and_completes(env: Path, session: Session) -> None:
    """No labels does not mean no judge - the selection check is a real call.

    `--no-cache` because the metastore is shared across the integration suite:
    another test may already have paid for these exact prompts, and a cache hit
    costing nothing is the correct behaviour, not the one under test here.
    """
    _run(env, "--no-cache")
    session.expire_all()
    run = session.scalars(select(EvalRun).order_by(EvalRun.started_at.desc()).limit(1)).one()
    assert run.status == "completed"
    assert run.cost_usd > 0
