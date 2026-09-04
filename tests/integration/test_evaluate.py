"""ingest -> evaluate against real Postgres, with a mocked judge.

The P0.7 acceptance criteria live here: every result row carries its evaluator
version, judged rows carry a judge hash and deterministic rows do not, and
traces with no ground truth land as `passed IS NULL` rather than false.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from sqlalchemy import Engine, func, select, text
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from evalloop.cli.main import app
from evalloop.store.models import EvalResultRow, EvalRun, JudgeConfigRow, LLMCache

pytestmark = pytest.mark.integration

runner = CliRunner()

EXAMPLE = Path("examples/support-bot")


@pytest.fixture
def project_dir(tmp_path: Path, unique_name: str) -> Path:
    """A copy of the shipped example, renamed and pointed at a mock judge.

    The real files, so the test fails if the example breaks - but with `provider:
    mock`, so CI needs no API key and the answers are deterministic.
    """
    for name in ("project.yaml", "judges.yaml", "eval-suite.yaml", "traces.jsonl"):
        shutil.copy(EXAMPLE / name, tmp_path / name)

    project = (tmp_path / "project.yaml").read_text(encoding="utf-8")
    (tmp_path / "project.yaml").write_text(
        project.replace("name: support-bot", f"name: {unique_name}"), encoding="utf-8"
    )
    judges = (tmp_path / "judges.yaml").read_text(encoding="utf-8")
    (tmp_path / "judges.yaml").write_text(
        judges.replace("provider: anthropic", "provider: mock"), encoding="utf-8"
    )
    return tmp_path


@pytest.fixture
def env(pg_engine: Engine, project_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("EVALLOOP_DATABASE_URL", pg_engine.url.render_as_string(False))
    monkeypatch.setenv("EVALLOOP_ARTIFACT_ROOT", str(project_dir / "artifacts"))
    return project_dir


@pytest.fixture
def session(pg_engine: Engine) -> Session:
    return sessionmaker(bind=pg_engine, expire_on_commit=False)()


def _ingest(directory: Path) -> None:
    result = runner.invoke(app, ["ingest", str(directory / "project.yaml")])
    assert result.exit_code == 0, result.output


def _evaluate(directory: Path, *args: str) -> str:
    result = runner.invoke(app, ["evaluate", str(directory / "eval-suite.yaml"), *args])
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def _run_id(session: Session) -> str:
    session.expire_all()
    return session.scalars(select(EvalRun.id).order_by(EvalRun.started_at.desc()).limit(1)).one()


def _judge_hashes(session: Session, run_id: str) -> set[str]:
    """Judge hashes used by one run.

    Scoped deliberately. The metastore is shared across the integration suite,
    so asserting against a global count of judge_config rows makes a test fail
    whenever an unrelated one inserts a row - which is a property of the test,
    not of the code.
    """
    return {
        value
        for value in session.scalars(
            select(EvalResultRow.judge_config_hash).where(
                EvalResultRow.run_id == run_id,
                EvalResultRow.judge_config_hash.is_not(None),
            )
        ).all()
        if value is not None
    }


def test_the_full_chain_writes_one_row_per_trace_and_evaluator(env: Path, session: Session) -> None:
    _ingest(env)
    _evaluate(env, "--split", "train")

    run_id = _run_id(session)
    count = session.scalar(
        select(func.count()).select_from(EvalResultRow).where(EvalResultRow.run_id == run_id)
    )
    assert count == 14 * 4  # 14 traces, 4 evaluators


def test_every_row_records_its_evaluator_version(env: Path, session: Session) -> None:
    """Rule 3. Without it, "the metric moved" and "the check changed underneath
    me" are indistinguishable."""
    _ingest(env)
    _evaluate(env, "--split", "train")

    missing = session.scalar(
        select(func.count())
        .select_from(EvalResultRow)
        .where(EvalResultRow.evaluator_version.is_(None))
    )
    assert missing == 0


def test_judged_rows_carry_a_judge_hash_and_deterministic_rows_do_not(
    env: Path, session: Session
) -> None:
    """A deterministic check has no judge at all, which is what makes it
    admissible as a gate floor the training loop could not influence."""
    _ingest(env)
    _evaluate(env, "--split", "train")
    run_id = _run_id(session)

    rows = session.execute(
        select(EvalResultRow.evaluator_id, EvalResultRow.judge_config_hash).where(
            EvalResultRow.run_id == run_id
        )
    ).all()

    by_id: dict[str, set[str | None]] = {}
    for evaluator_id, judge_hash in rows:
        by_id.setdefault(evaluator_id, set()).add(judge_hash)

    assert by_id["tool_call_match"] == {None}
    assert by_id["tool_name_match"] == {None}
    assert None not in by_id["policy_followed"]
    assert None not in by_id["escalation_correct"]


def test_the_judge_config_is_registered_under_its_hash(env: Path, session: Session) -> None:
    """eval_result has a foreign key onto judge_config.hash: rule 2 enforced by
    the schema, so a result cannot exist without its judge on record."""
    _ingest(env)
    _evaluate(env, "--split", "train")

    used = _judge_hashes(session, _run_id(session))
    assert len(used) == 2  # two questions, two different judge versions

    for judge_hash in used:
        assert session.get(JudgeConfigRow, judge_hash) is not None


def test_traces_without_ground_truth_are_null_not_false(env: Path, session: Session) -> None:
    """The decision this whole design turns on. `false` would be a verdict the
    data does not support, and P4 would compile it into training data."""
    _ingest(env)
    _evaluate(env, "--split", "train")
    run_id = _run_id(session)

    rows = session.execute(
        text(
            "SELECT passed, count(*) FROM eval_result "
            "WHERE run_id = :r AND evaluator_id = 'tool_call_match' "
            "GROUP BY passed ORDER BY passed NULLS LAST"
        ),
        {"r": run_id},
    ).all()

    tally = {passed: count for passed, count in rows}
    assert tally[True] == 5
    assert tally[False] == 5
    assert tally[None] == 4  # four traces have no expected tool calls


def test_a_holdout_question_is_measured_but_never_scored(env: Path, session: Session) -> None:
    """No ground truth by design, so every row is null. The answer is still
    recorded - that is what makes it usable at the gate."""
    _ingest(env)
    _evaluate(env, "--split", "train")
    run_id = _run_id(session)

    rows = session.execute(
        select(EvalResultRow.passed, EvalResultRow.normalized_prediction).where(
            EvalResultRow.run_id == run_id,
            EvalResultRow.evaluator_id == "escalation_correct",
        )
    ).all()

    assert len(rows) == 14
    assert all(passed is None for passed, _ in rows)
    assert all(prediction is not None for _, prediction in rows)


def test_cost_and_tokens_roll_up_onto_the_run(env: Path, session: Session) -> None:
    _ingest(env)
    _evaluate(env, "--split", "train", "--no-cache")

    run = session.get(EvalRun, _run_id(session))
    assert run is not None
    assert run.cost_usd > 0
    assert run.tokens_in > 0
    assert run.status == "completed"


def test_a_second_run_is_served_from_cache_and_costs_nothing(env: Path, session: Session) -> None:
    """Keyed by judge version, so a rubric edit could never reuse these."""
    _ingest(env)
    _evaluate(env, "--split", "train", "--no-cache")
    output = _evaluate(env, "--split", "train")

    assert "cache hits 28" in output  # 14 traces x 2 judged questions
    run = session.get(EvalRun, _run_id(session))
    assert run is not None
    assert run.cost_usd == 0.0
    assert run.cache_hits == 28


def test_cache_entries_are_keyed_by_judge_version(env: Path, session: Session) -> None:
    _ingest(env)
    _evaluate(env, "--split", "train", "--no-cache")

    used = _judge_hashes(session, _run_id(session))
    cached = session.scalars(
        select(LLMCache.judge_config_hash).where(LLMCache.judge_config_hash.in_(used))
    ).all()

    assert len(cached) == 28  # 14 traces x 2 judged questions
    assert set(cached) == used


def test_editing_the_rubric_changes_the_judge_hash_and_misses_the_cache(
    env: Path, session: Session
) -> None:
    """The single most important property of the cache: a new question can never
    be answered with the old question's reply."""
    _ingest(env)
    _evaluate(env, "--split", "train")
    before = _judge_hashes(session, _run_id(session))

    suite_path = env / "eval-suite.yaml"
    suite_path.write_text(
        suite_path.read_text(encoding="utf-8").replace(
            "Did the agent follow the refund policy?",
            "Did the agent comply with the refund policy?",
        ),
        encoding="utf-8",
    )

    _evaluate(env, "--split", "train")
    after = _judge_hashes(session, _run_id(session))

    # The edited question is a new judge version; the untouched one is not.
    new_hashes = after - before
    assert len(new_hashes) == 1
    assert len(before & after) == 1

    # And the new version has its own cache namespace, disjoint from the old
    # one. This is the property that makes recalibrating a prompt safe: the new
    # question cannot be answered with the old question's reply.
    old_keys = set(
        session.scalars(
            select(LLMCache.key).where(LLMCache.judge_config_hash.in_(before - after))
        ).all()
    )
    new_keys = set(
        session.scalars(
            select(LLMCache.key).where(LLMCache.judge_config_hash.in_(new_hashes))
        ).all()
    )
    assert len(old_keys) == 14
    assert len(new_keys) == 14
    assert old_keys.isdisjoint(new_keys)


def test_the_summary_reports_coverage_separately_from_failure(env: Path) -> None:
    _ingest(env)
    output = _evaluate(env, "--split", "train")
    assert "nothing to compare against" in output
    assert "Not counted as failures" in output


def test_limit_restricts_the_traces_evaluated(env: Path, session: Session) -> None:
    _ingest(env)
    _evaluate(env, "--split", "train", "--limit", "3")

    count = session.scalar(
        select(func.count())
        .select_from(EvalResultRow)
        .where(EvalResultRow.run_id == _run_id(session))
    )
    assert count == 12  # 3 traces x 4 evaluators


def test_evaluating_before_ingesting_says_so(env: Path) -> None:
    result = runner.invoke(app, ["evaluate", str(env / "eval-suite.yaml")])
    assert result.exit_code == 1
    assert "ingest" in result.output


def test_a_missing_sibling_config_is_reported(env: Path) -> None:
    """evaluate discovers judges.yaml beside the suite. Saying which file is
    missing beats a KeyError about an undeclared judge."""
    (env / "judges.yaml").unlink()
    result = runner.invoke(app, ["evaluate", str(env / "eval-suite.yaml")])
    assert result.exit_code == 1
    assert "judges.yaml" in result.output


def test_an_unimplemented_evaluator_type_is_refused_by_name(env: Path) -> None:
    suite_path = env / "eval-suite.yaml"
    suite_path.write_text(
        suite_path.read_text(encoding="utf-8").replace("type: json_match", "type: regex"),
        encoding="utf-8",
    )
    _ingest(env)
    result = runner.invoke(app, ["evaluate", str(suite_path)])
    assert result.exit_code == 1
    assert "not implemented yet" in " ".join(result.output.split())


def test_an_unknown_split_is_a_clear_error(env: Path) -> None:
    _ingest(env)
    result = runner.invoke(app, ["evaluate", str(env / "eval-suite.yaml"), "--split", "dev"])
    assert result.exit_code == 1
    assert "no traces in split" in " ".join(result.output.split())


def test_explicit_snapshot_is_honoured(env: Path, session: Session) -> None:
    _ingest(env)
    snapshot_id = session.scalars(
        text("SELECT id FROM snapshot ORDER BY created_at DESC LIMIT 1")
    ).one()

    output = _evaluate(env, "--split", "train", "--snapshot", snapshot_id)
    assert snapshot_id in output


def test_judged_rows_keep_the_reason_the_judge_gave(env: Path, session: Session) -> None:
    """The explanation is what a human reads when auditing a disagreement, so
    losing it makes the judgecard unusable for anything but counting."""
    _ingest(env)
    _evaluate(env, "--split", "train")

    explanation = session.scalars(
        select(EvalResultRow.explanation).where(
            EvalResultRow.evaluator_id == "policy_followed",
            EvalResultRow.explanation.is_not(None),
        )
    ).first()
    assert explanation
