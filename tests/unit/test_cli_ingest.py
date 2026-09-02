"""The ingest command's surface: config errors, exit codes, dry-run output."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from evalloop.cli.main import app

runner = CliRunner()

ROW = {
    "id": "sb-1",
    "user_transcript": "refund please",
    "assistant_transcript": "done",
    "language": "en",
}

PROJECT = """name: cli-test
source:
  type: jsonl
  path: traces.jsonl
mapping:
  trace_id: id
  input.user_request: user_transcript
  output.text: assistant_transcript
  metadata.language: language
"""


def _project_dir(tmp_path: Path, rows: list[dict[str, object]] | None = None) -> Path:
    (tmp_path / "project.yaml").write_text(PROJECT, encoding="utf-8")
    with (tmp_path / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows if rows is not None else [ROW, {**ROW, "id": "sb-2"}]:
            handle.write(json.dumps(row) + "\n")
    return tmp_path


def test_invalid_project_file_is_reported_and_never_reaches_the_database(
    tmp_path: Path,
) -> None:
    """A config error must fail before any connection is attempted, so the
    message is about the typo rather than about Postgres being unreachable."""
    (tmp_path / "project.yaml").write_text(PROJECT.replace("  path:", "  pth:"), encoding="utf-8")
    result = runner.invoke(app, ["ingest", str(tmp_path / "project.yaml")])

    assert result.exit_code == 1
    output = " ".join(result.output.split())
    assert "unknown key" in output or "requires source.path" in output


def test_missing_project_file(tmp_path: Path) -> None:
    result = runner.invoke(app, ["ingest", str(tmp_path / "absent.yaml")])
    assert result.exit_code == 1
    assert "file not found" in result.output


def test_dry_run_reports_without_touching_the_database(tmp_path: Path) -> None:
    directory = _project_dir(tmp_path)
    result = runner.invoke(
        app, ["ingest", str(directory / "project.yaml"), "--dry-run", "--limit", "1"]
    )

    assert result.exit_code == 0
    output = " ".join(result.output.split())
    assert "dry run" in output
    assert "nothing written" in output
    assert "cli-test" in output
    assert "sb-1" in output  # the first mapped trace is shown


def test_dry_run_surfaces_unmapped_source_fields(tmp_path: Path) -> None:
    """A forgotten mapping line is otherwise invisible until someone wonders
    why a slice came back empty."""
    directory = _project_dir(tmp_path, [{**ROW, "tier": "premium", "region": "APAC"}])
    result = runner.invoke(app, ["ingest", str(directory / "project.yaml"), "--dry-run"])

    output = " ".join(result.output.split())
    assert "no mapping" in output
    assert "region" in output and "tier" in output


def test_dry_run_reports_bad_source_lines(tmp_path: Path) -> None:
    directory = _project_dir(tmp_path)
    with (directory / "traces.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")

    result = runner.invoke(app, ["ingest", str(directory / "project.yaml"), "--dry-run"])
    assert result.exit_code == 1
    assert "invalid JSON" in " ".join(result.output.split())


def test_empty_source_fails(tmp_path: Path) -> None:
    """An empty snapshot would evaluate to a perfect score on zero traces."""
    directory = _project_dir(tmp_path, [])
    result = runner.invoke(app, ["ingest", str(directory / "project.yaml"), "--dry-run"])
    assert result.exit_code == 1
    assert "no traces" in result.output


def test_ingest_appears_in_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert "ingest" in result.output
    assert "snapshot" in result.output


def test_many_bad_lines_are_truncated_in_the_output(tmp_path: Path) -> None:
    """Thirty malformed lines should not push the summary off the screen."""
    directory = _project_dir(tmp_path)
    with (directory / "traces.jsonl").open("a", encoding="utf-8") as handle:
        for _ in range(30):
            handle.write("{not json\n")

    result = runner.invoke(app, ["ingest", str(directory / "project.yaml"), "--dry-run"])
    output = " ".join(result.output.split())
    assert "and 10 more" in output


def test_skipped_rows_are_counted_in_the_summary(tmp_path: Path) -> None:
    """A row that parses as JSON but cannot be mapped is a different failure
    from a syntax error, and is reported separately."""
    directory = _project_dir(tmp_path, [ROW, {"user_transcript": "no id"}])
    result = runner.invoke(app, ["ingest", str(directory / "project.yaml"), "--dry-run"])
    output = " ".join(result.output.split())
    assert "skipped 1" in output
    assert "trace_id" in output
