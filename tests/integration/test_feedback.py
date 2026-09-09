"""`evalloop feedback build` against a real run.

The unit tests pin the compiling. This pins the joins: a training row needs the
result *and* the trace, from two different stores, plus the split assignment
that keeps the sealed set out.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from sqlalchemy import Engine
from typer.testing import CliRunner

from evalloop.cli.main import app

pytestmark = pytest.mark.integration

runner = CliRunner()

EXAMPLE = Path("examples/support-bot")


@pytest.fixture
def env(
    pg_engine: Engine, tmp_path: Path, unique_name: str, monkeypatch: pytest.MonkeyPatch
) -> Path:
    for name in ("project.yaml", "judges.yaml", "eval-suite.yaml", "tools.yaml", "traces.jsonl"):
        shutil.copy(EXAMPLE / name, tmp_path / name)

    for name, old, new in (
        ("project.yaml", "name: support-bot", f"name: {unique_name}"),
        ("judges.yaml", "provider: anthropic", "provider: mock"),
        # Its own judge namespace, so the cache-count assertions in
        # test_evaluate.py stay meaningful across the shared metastore.
        ("judges.yaml", "model: claude-sonnet-5", "model: stub-feedback"),
    ):
        path = tmp_path / name
        path.write_text(path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")

    monkeypatch.setenv("EVALLOOP_DATABASE_URL", pg_engine.url.render_as_string(False))
    monkeypatch.setenv("EVALLOOP_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    return tmp_path


def _evaluated(env: Path) -> None:
    assert runner.invoke(app, ["ingest", str(env / "project.yaml")]).exit_code == 0
    result = runner.invoke(app, ["evaluate", str(env / "eval-suite.yaml"), "--split", "train"])
    assert result.exit_code == 0, result.output


def _build(env: Path, *args: str) -> tuple[int, str, Path]:
    out = env / "feedback.jsonl"
    result = runner.invoke(app, ["feedback", "build", "--out", str(out), *args])
    return result.exit_code, " ".join(result.output.split()), out


def test_it_compiles_a_run_into_a_dataset_file(env: Path) -> None:
    _evaluated(env)
    code, output, out = _build(env, "--tools", str(env / "tools.yaml"))

    # The mock judge proposes empty arguments, which the registry rejects, so
    # this run legitimately emits nothing - and says why.
    assert code in (0, 2), output
    assert out.exists()
    assert "preference pair" in output
    assert "Dropped" in output


def test_the_drop_reasons_are_reported_rather_than_swallowed(env: Path) -> None:
    _evaluated(env)
    _, output, _ = _build(env, "--tools", str(env / "tools.yaml"))
    assert "proposal_failed_argument_check" in output or "passed" in output


def test_every_emitted_row_is_valid_jsonl_with_provenance(env: Path) -> None:
    _evaluated(env)
    _, _, out = _build(env, "--tools", str(env / "tools.yaml"))
    for line in out.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert {"prompt", "chosen", "rejected", "target_source", "judge_version"} <= set(row)
        assert row["signal_provenance"] == "judge"


def test_without_a_registry_no_proposal_is_trusted(env: Path) -> None:
    """A judge-proposed call that no deterministic check validated never becomes
    training data."""
    _evaluated(env)
    _, output, _ = _build(env)
    assert "no_registry_to_validate_proposal" in output


def test_an_unimplemented_strategy_is_refused_by_name(env: Path) -> None:
    _evaluated(env)
    result = runner.invoke(app, ["feedback", "build", "--strategy", "sft"])
    assert result.exit_code == 1
    assert "not implemented" in result.output


def test_a_run_without_the_selection_check_says_what_is_missing(env: Path) -> None:
    _evaluated(env)
    result = runner.invoke(app, ["feedback", "build", "--selection", "absent"])
    assert result.exit_code == 1
    assert "no 'absent' results" in " ".join(result.output.split())


def test_an_unknown_run_is_named(env: Path) -> None:
    _evaluated(env)
    result = runner.invoke(app, ["feedback", "build", "run-does-not-exist"])
    assert result.exit_code == 1
    assert "no such run" in result.output
