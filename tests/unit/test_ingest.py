"""The JSONL connector, the mapping engine, and Parquet round-trips."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from evalloop.contracts import ProjectConfig, Trace
from evalloop.ingest.connectors.jsonl import JsonlConnector
from evalloop.ingest.mapping import apply_mapping, map_row
from evalloop.ingest.pipeline import build_traces
from evalloop.store import LocalArtifactStore, read_traces, write_traces

SOURCE_ROW = {
    "id": "sb-0417",
    "user_transcript": "I want a refund",
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


def _jsonl(tmp_path: Path, rows: list[Any], name: str = "traces.jsonl") -> Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


# --- connector ---


def test_reads_one_row_per_line(tmp_path: Path) -> None:
    connector = JsonlConnector(path=_jsonl(tmp_path, [SOURCE_ROW, SOURCE_ROW]))
    rows = list(connector.rows())
    assert len(rows) == 2
    assert connector.errors == []


def test_source_id_records_the_line_number(tmp_path: Path) -> None:
    """A mapping problem has to be traceable back to the exact input row."""
    connector = JsonlConnector(path=_jsonl(tmp_path, [SOURCE_ROW, SOURCE_ROW]))
    assert [r.source_id for r in connector.rows()] == ["line:1", "line:2"]


def test_bad_lines_are_collected_not_raised(tmp_path: Path) -> None:
    """A 100k-line export with three bad rows should ingest 99,997 traces and
    tell you about the three, not stop at line 2."""
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        json.dumps(SOURCE_ROW) + "\n" + "{not json\n" + json.dumps(SOURCE_ROW) + "\n",
        encoding="utf-8",
    )
    connector = JsonlConnector(path=path)
    rows = list(connector.rows())
    assert len(rows) == 2
    assert len(connector.errors) == 1
    assert "line 2" in connector.errors[0]


def test_non_object_lines_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "arrays.jsonl"
    path.write_text('[1, 2, 3]\n{"id": "x"}\n', encoding="utf-8")
    connector = JsonlConnector(path=path)
    assert len(list(connector.rows())) == 1
    assert "expected an object" in connector.errors[0]


def test_blank_and_commented_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "spaced.jsonl"
    path.write_text(f"\n// a note\n{json.dumps(SOURCE_ROW)}\n\n", encoding="utf-8")
    connector = JsonlConnector(path=path)
    assert len(list(connector.rows())) == 1
    assert connector.errors == []


def test_limit_stops_early(tmp_path: Path) -> None:
    connector = JsonlConnector(path=_jsonl(tmp_path, [SOURCE_ROW] * 10))
    assert len(list(connector.rows(limit=3))) == 3


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(JsonlConnector(path=tmp_path / "absent.jsonl").rows())


def test_fingerprint_config_uses_the_resolved_path(tmp_path: Path) -> None:
    connector = JsonlConnector(path=_jsonl(tmp_path, [SOURCE_ROW]))
    config = connector.fingerprint_config()
    assert config["type"] == "jsonl"
    assert Path(config["path"]).is_absolute()


# --- mapping ---


def test_maps_source_columns_onto_trace_paths() -> None:
    trace = map_row(SOURCE_ROW, MAPPING, source_id="line:1")
    assert trace.trace_id == "sb-0417"
    assert trace.input.user_request == "I want a refund"
    assert trace.output.tool_calls[0].name == "issue_refund"
    assert trace.ground_truth.get("policy_followed") is False
    assert trace.metadata == {"language": "en"}
    assert trace.source_id == "line:1"


def test_empty_mapping_treats_the_row_as_already_shaped() -> None:
    """The common case for a JSONL export produced for EvalLoop rather than
    harvested from a product database."""
    native = {"trace_id": "t1", "input": {"user_request": "hi"}, "output": {"text": "yo"}}
    trace = map_row(native, {})
    assert trace.trace_id == "t1"
    assert trace.output.text == "yo"


def test_absent_source_column_is_left_unset_not_nulled() -> None:
    """An optional column that is simply missing must not become an explicit
    null - `ground_truth.tool_calls: null` is a claim that no tool should have
    been called, which is different from having no opinion."""
    row = {k: v for k, v in SOURCE_ROW.items() if k != "expected_reply"}
    trace = map_row(row, MAPPING)
    assert not trace.ground_truth.has("expected_response")
    assert trace.ground_truth.has("policy_followed")


def test_a_row_with_no_ground_truth_columns_is_still_valid() -> None:
    row = {"id": "t1", "user_transcript": "hi", "assistant_transcript": "yo"}
    trace = map_row(row, MAPPING)
    assert trace.ground_truth.is_empty


def test_input_and_output_always_exist_even_when_unmapped() -> None:
    trace = map_row({"id": "t1"}, {"trace_id": "id"})
    assert trace.input.user_request is None
    assert trace.output.tool_calls == []


def test_unmappable_row_is_skipped_and_reported() -> None:
    """One bad row out of ten thousand should cost that row and a line of
    output, not the whole ingest."""
    rows = [(SOURCE_ROW, "line:1"), ({"user_transcript": "no id here"}, "line:2")]
    result = apply_mapping(rows, MAPPING)
    assert len(result.traces) == 1
    assert result.skipped == 1
    assert "line:2" in result.errors[0]
    assert "trace_id" in result.errors[0]


def test_unmapped_source_fields_are_reported() -> None:
    """A forgotten mapping line is otherwise invisible until someone wonders
    why a slice came back empty."""
    row = {**SOURCE_ROW, "tier": "premium", "region": "APAC"}
    result = apply_mapping([(row, "line:1")], MAPPING)
    assert result.unmapped_fields == {"tier", "region"}


def test_native_rows_report_no_unmapped_fields() -> None:
    native = {"trace_id": "t1", "input": {}, "output": {}}
    assert apply_mapping([(native, "line:1")], {}).unmapped_fields == set()


def test_mapping_into_a_list_index() -> None:
    """The voice case: the recording lives in the customer's own column and the
    trace carries a pointer to it."""
    row = {"id": "call-1", "recording_url": "s3://bucket/call-1.wav", "kind": "audio"}
    trace = map_row(
        row,
        {
            "trace_id": "id",
            "output.artifacts[0].uri": "recording_url",
            "output.artifacts[0].type": "kind",
        },
    )
    assert trace.output.artifacts[0].uri == "s3://bucket/call-1.wav"
    assert trace.output.artifacts[0].type == "audio"


def test_partially_mapped_list_element_fails_loudly() -> None:
    """Mapping `uri` but forgetting `type` builds half an artifact. That is a
    mapping bug, and a trace carrying a pointer with no declared kind is worse
    than an error - something downstream would try to fetch it."""
    row = {"id": "call-1", "recording_url": "s3://bucket/call-1.wav"}
    with pytest.raises(ValidationError, match="type"):
        map_row(row, {"trace_id": "id", "output.artifacts[0].uri": "recording_url"})


def test_a_bad_value_fails_validation_rather_than_being_coerced() -> None:
    with pytest.raises(ValidationError):
        map_row(
            {"id": "t1", "calls": "not a list"}, {"trace_id": "id", "output.tool_calls": "calls"}
        )


# --- parquet round trip ---


def _traces(n: int = 3) -> list[Trace]:
    return [map_row({**SOURCE_ROW, "id": f"t{i}"}, MAPPING) for i in range(n)]


def test_parquet_round_trip_is_lossless(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    original = _traces()
    restored = read_traces(store, write_traces(store, original))

    assert [t.trace_id for t in restored] == [t.trace_id for t in original]
    assert restored[0].output.tool_calls[0].arguments == {"order_id": "ORD-1"}
    assert restored[0].ground_truth.get("policy_followed") is False
    assert restored[0].content_hash == original[0].content_hash


def test_identical_traces_produce_one_artifact(tmp_path: Path) -> None:
    """Content addressing only dedups if the bytes are deterministic. A per-trace
    `ingested_at` in the payload broke this - identical data wrote a second file
    differing only by a timestamp."""
    store = LocalArtifactStore(tmp_path)
    traces = _traces()
    assert write_traces(store, traces) == write_traces(store, list(traces))
    assert sum(1 for p in tmp_path.rglob("*") if p.is_file()) == 1


def test_content_hash_mismatch_is_detected(tmp_path: Path) -> None:
    """The recorded hash is the claim; recomputing it on read is the check.
    Handing back a trace that quietly is not the one ingested is worse than
    failing."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from evalloop.store.traces import SCHEMA

    store = LocalArtifactStore(tmp_path)
    trace = _traces(1)[0]
    tampered = trace.model_copy(update={"trace_id": "someone-else"})

    table = pa.Table.from_pydict(
        {
            "trace_id": [trace.trace_id],
            "split": ["train"],
            "content_hash": [trace.content_hash],
            "payload": [tampered.model_dump_json(exclude={"ingested_at", "content_hash"})],
        },
        schema=SCHEMA,
    )
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer)
    uri = store.put_bytes(buffer.getvalue().to_pybytes())

    with pytest.raises(ValueError, match="does not match its recorded content hash"):
        read_traces(store, uri)
    assert read_traces(store, uri, verify=False)[0].trace_id == "someone-else"


def test_empty_trace_list_round_trips(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    assert read_traces(store, write_traces(store, [])) == []


# --- build_traces ---


def _project(tmp_path: Path, **overrides: Any) -> ProjectConfig:
    return ProjectConfig.model_validate(
        {
            "name": "support-bot",
            "source": {"type": "jsonl", "path": "traces.jsonl"},
            "mapping": MAPPING,
            **overrides,
        }
    )


def test_build_traces_resolves_the_source_relative_to_the_project(tmp_path: Path) -> None:
    """So a project directory can be checked out anywhere and still work."""
    _jsonl(tmp_path, [SOURCE_ROW, {**SOURCE_ROW, "id": "sb-0418"}])
    traces, report = build_traces(_project(tmp_path), root=tmp_path)

    assert len(traces) == 2
    assert report.row_count == 2
    assert report.fingerprint
    assert report.ok


def test_build_traces_honours_limit(tmp_path: Path) -> None:
    _jsonl(tmp_path, [SOURCE_ROW] * 5)
    traces, _ = build_traces(_project(tmp_path), root=tmp_path, limit=2)
    assert len(traces) == 2


def test_fingerprint_is_stable_across_runs(tmp_path: Path) -> None:
    _jsonl(tmp_path, [SOURCE_ROW, {**SOURCE_ROW, "id": "sb-0418"}])
    project = _project(tmp_path)
    first = build_traces(project, root=tmp_path)[1].fingerprint
    second = build_traces(project, root=tmp_path)[1].fingerprint
    assert first == second


def test_fingerprint_changes_when_the_data_changes(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _jsonl(tmp_path, [SOURCE_ROW])
    before = build_traces(project, root=tmp_path)[1].fingerprint
    _jsonl(tmp_path, [SOURCE_ROW, {**SOURCE_ROW, "id": "sb-0418"}])
    assert build_traces(project, root=tmp_path)[1].fingerprint != before


def test_unimplemented_source_type_is_a_clear_error(tmp_path: Path) -> None:
    project = ProjectConfig.model_validate(
        {
            "name": "p",
            "source": {"type": "postgres", "query": "SELECT * FROM traces"},
            "mapping": MAPPING,
        }
    )
    traces, report = build_traces(project, root=tmp_path)
    assert traces == []
    assert "not implemented yet" in report.errors[0]
    assert not report.ok


def test_absolute_source_path_is_used_as_given(tmp_path: Path) -> None:
    path = _jsonl(tmp_path, [SOURCE_ROW], name="elsewhere.jsonl")
    project = _project(tmp_path, source={"type": "jsonl", "path": str(path)})
    traces, _ = build_traces(project, root=Path("/nonexistent"))
    assert len(traces) == 1


def test_conflicting_mapping_targets_cost_one_row_not_the_run() -> None:
    """Two targets that disagree about a container's shape - a list here, a
    mapping there - raise a TypeError. It still costs one row and one line, not
    a traceback that buries the other 9,999 rows."""
    rows = [({"id": "t1", "calls": [], "name": "cancel_order"}, "line:1")]
    result = apply_mapping(
        rows,
        {
            "trace_id": "id",
            "output.tool_calls": "calls",
            "output.tool_calls.name": "name",
        },
    )

    assert result.traces == []
    assert result.skipped == 1
    assert len(result.errors) == 1
    assert "\n" not in result.errors[0]
