"""`evalloop report tools` against a real run.

The unit tests pin the counting. This pins the wiring: that the columns the
report reads are the columns evaluation actually writes. The bug this exists to
prevent is silent — the report renders happily, with every row saying nothing
was called and nothing was expected.
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
    """The shipped example, with two traces edited to produce registry violations.

    The mock judge cannot manufacture an illegal call - it only ever answers the
    selection question - so the violations table needs traces that misbehave.
    """
    for name in ("project.yaml", "judges.yaml", "eval-suite.yaml", "tools.yaml"):
        shutil.copy(EXAMPLE / name, tmp_path / name)

    rows = [
        json.loads(line)
        for line in (EXAMPLE / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows[0]["tool_calls"] = [{"name": "refund_order_now", "arguments": {"order_id": "ORD-8891"}}]
    rows[4]["tool_calls"] = [
        {"name": "issue_refund", "arguments": {"order_id": "ORD-2210", "amount": 59.0}},
        {"name": "issue_refund", "arguments": {"order_id": "ORD-2210", "amount": 59.0}},
    ]
    (tmp_path / "traces.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )

    for name, old, new in (
        ("project.yaml", "name: support-bot", f"name: {unique_name}"),
        ("judges.yaml", "provider: anthropic", "provider: mock"),
        # A distinct model name gives this file its own judge version, and
        # therefore its own cache namespace. The metastore is shared across the
        # whole integration suite, and `llm_cache` rows are keyed by judge hash
        # with no run column - so a test that edits trace content writes cache
        # rows that another test's global count would pick up. Editing traces is
        # exactly what this fixture does.
        ("judges.yaml", "model: claude-sonnet-5", "model: stub-report"),
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


def _report(*args: str) -> str:
    result = runner.invoke(app, ["report", "tools", *args])
    assert result.exit_code == 0, result.output
    return " ".join(result.output.split())


def test_the_report_names_what_was_called_and_what_the_judge_chose(env: Path) -> None:
    _evaluated(env)
    output = _report()

    assert "Wrong tool selections" in output
    # The mock judge always picks the alphabetically first tool in the
    # catalogue, so every disagreement points at cancel_order.
    assert "issue_refund cancel_order" in output
    # sb-0421 called the right tool and then also refunded.
    assert "cancel_order,issue_refund" in output


def test_predictions_survive_the_round_trip_through_postgres(env: Path) -> None:
    """The silent failure this guards: `normalized_prediction` is stored inside
    a `{"value": ...}` envelope, and reading the envelope makes every row say
    `none → none`."""
    _evaluated(env)
    output = _report()
    assert "none none" not in output


def test_registry_violations_are_grouped_by_code(env: Path) -> None:
    _evaluated(env)
    output = _report()

    assert "Registry violations" in output
    assert "unregistered_tool refund_order_now" in output
    assert "duplicate_side_effecting issue_refund" in output


def test_coverage_is_reported_beside_the_tables(env: Path) -> None:
    """Traces that called nothing are not failures and not passes."""
    _evaluated(env)
    output = _report()
    assert "no tool calls 4" in output
    assert "agreed" in output


def test_text_consistency_is_reported_as_its_own_section(env: Path) -> None:
    """The third axis. The mock judge answers `consistent: true` from the
    schema, so this exercises the clean branch; the failing branch is pinned in
    the unit tests, where the answer can be dictated."""
    _evaluated(env)
    output = _report()
    assert "No reply contradicted its calls" in output
    assert "consistent 14" in output


def test_it_defaults_to_the_most_recent_run(env: Path) -> None:
    _evaluated(env)
    assert "Wrong tool selections" in _report()


def test_an_unknown_run_is_named_rather_than_rendered_empty(env: Path) -> None:
    _evaluated(env)
    result = runner.invoke(app, ["report", "tools", "run-does-not-exist"])
    assert result.exit_code == 1
    assert "no such run" in result.output


def test_a_run_with_no_tool_checks_says_so(env: Path) -> None:
    """Distinct from finding nothing wrong, and the message names what it looked
    for so a renamed evaluator id is diagnosable."""
    _evaluated(env)
    missing = ["--selection", "absent", "--registry", "gone", "--consistency", "missing"]
    result = runner.invoke(app, ["report", "tools", *missing])
    assert result.exit_code == 1
    assert "no tool checks" in " ".join(result.output.split())
