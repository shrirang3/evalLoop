"""The validate command: exit codes and rendering."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalloop.cli.main import app

runner = CliRunner()

GOOD_SUITE = """suite: support-bot-v1
evaluators:
  - id: tool_call_match
    type: json_match
    actual: output.tool_calls
    expected: ground_truth.tool_calls
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_file_exits_zero(tmp_path: Path) -> None:
    path = _write(tmp_path, "eval-suite.yaml", GOOD_SUITE)
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 0


def test_invalid_file_exits_one(tmp_path: Path) -> None:
    """The exit code is what CI reads, so it has to be right even when nobody is
    looking at the output."""
    path = _write(tmp_path, "eval-suite.yaml", GOOD_SUITE.replace("expected:", "expcted:"))
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 1


def test_quiet_prints_nothing_but_still_fails(tmp_path: Path) -> None:
    path = _write(tmp_path, "eval-suite.yaml", GOOD_SUITE.replace("expected:", "expcted:"))
    result = runner.invoke(app, ["validate", "--quiet", str(path)])
    assert result.exit_code == 1
    assert result.output.strip() == ""


def test_output_carries_line_snippet_and_hint(tmp_path: Path) -> None:
    path = _write(tmp_path, "eval-suite.yaml", GOOD_SUITE.replace("expected:", "expcted:"))
    result = runner.invoke(app, ["validate", str(path)])
    output = " ".join(result.output.split())

    assert "unknown key" in output
    assert ":6" in output
    assert "expcted" in output
    assert "did you mean 'expected'?" in output
    assert "^" in output
    assert "1 problem found" in output


def test_as_flag_forces_a_schema(tmp_path: Path) -> None:
    path = _write(tmp_path, "mystery.yaml", GOOD_SUITE)
    assert runner.invoke(app, ["validate", str(path)]).exit_code == 0  # sniffed
    assert runner.invoke(app, ["validate", "--as", "eval-suite", str(path)]).exit_code == 0
    assert runner.invoke(app, ["validate", "--as", "project", str(path)]).exit_code == 1


def test_several_files_are_all_reported(tmp_path: Path) -> None:
    good = _write(tmp_path, "eval-suite.yaml", GOOD_SUITE)
    bad = _write(tmp_path, "promotion.yaml", "slices: []\n")
    result = runner.invoke(app, ["validate", str(good), str(bad)])
    assert result.exit_code == 1
    assert "promotion.yaml" in result.output


def test_valid_run_names_each_file_and_its_kind(tmp_path: Path) -> None:
    path = _write(tmp_path, "eval-suite.yaml", GOOD_SUITE)
    result = runner.invoke(app, ["validate", str(path)])
    output = " ".join(result.output.split())
    assert "eval-suite" in output
    assert "1 file valid" in output


@pytest.mark.parametrize("flag", ["--version", "-V"])
def test_version_flag(flag: str) -> None:
    result = runner.invoke(app, [flag])
    assert result.exit_code == 0
    assert "evalloop" in result.output


def test_long_line_is_truncated_so_the_caret_stays_aligned(tmp_path: Path) -> None:
    """Wrapping would move the caret away from the token it points at."""
    long_value = "x" * 400
    text = GOOD_SUITE.replace("expected: ground_truth.tool_calls", f"expcted: {long_value}")
    path = _write(tmp_path, "eval-suite.yaml", text)
    result = runner.invoke(app, ["validate", str(path)])
    assert result.exit_code == 1
    assert "…" in result.output


def test_shipped_example_configs_all_validate() -> None:
    """The examples are documentation people copy. A broken one teaches a broken
    pattern, so they are checked like code."""
    examples = sorted(Path("examples/support-bot").glob("*.yaml"))
    assert len(examples) == 5, "expected project, judges, eval-suite, promotion, training"

    result = runner.invoke(app, ["validate", *[str(p) for p in examples]])
    assert result.exit_code == 0, result.output
    assert "5 files valid" in " ".join(result.output.split())
