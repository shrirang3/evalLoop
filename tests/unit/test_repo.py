"""Repository behaviour that carries a rule: snapshot idempotency, judge config
registration, and cost roll-up."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from evalloop.contracts import EvalResult, Trace
from evalloop.store import (
    EvalResultRow,
    SplitAssignment,
    TraceRow,
    record_results,
    source_fingerprint,
    start_run,
    upsert_judge_config,
    upsert_project,
    upsert_snapshot,
)

CONNECTOR = {"type": "jsonl", "path": "traces.jsonl"}


def _traces(n: int = 3) -> list[Trace]:
    return [
        Trace.model_validate(
            {
                "trace_id": f"t{i}",
                "input": {"user_request": f"request {i}"},
                "output": {"text": f"reply {i}"},
            }
        )
        for i in range(n)
    ]


def test_fingerprint_ignores_source_row_order() -> None:
    """A query returning the same rows in a different order is the same data,
    and must not produce a second snapshot."""
    hashes = [t.content_hash for t in _traces()]
    a = source_fingerprint(connector_config=CONNECTOR, content_hashes=hashes)
    b = source_fingerprint(connector_config=CONNECTOR, content_hashes=list(reversed(hashes)))
    assert a == b


def test_fingerprint_changes_with_content() -> None:
    hashes = [t.content_hash for t in _traces()]
    a = source_fingerprint(connector_config=CONNECTOR, content_hashes=hashes)
    b = source_fingerprint(connector_config=CONNECTOR, content_hashes=hashes[:-1])
    assert a != b


def test_reingesting_identical_data_does_not_duplicate(session: Session) -> None:
    """The failure this prevents is silent: two snapshots of the same 500 traces
    make every rate computed across them wrong, with nothing to notice."""
    project = upsert_project(session, name="support-bot", config_yaml="name: support-bot")
    traces = _traces()
    fingerprint = source_fingerprint(
        connector_config=CONNECTOR, content_hashes=[t.content_hash for t in traces]
    )

    first, created_first = upsert_snapshot(
        session, project_id=project.id, fingerprint=fingerprint, traces=traces
    )
    second, created_second = upsert_snapshot(
        session, project_id=project.id, fingerprint=fingerprint, traces=traces
    )

    assert created_first is True
    assert created_second is False
    assert first.id == second.id
    assert session.scalar(select(func.count()).select_from(TraceRow)) == 3


def test_snapshot_writes_one_split_assignment_per_trace(session: Session) -> None:
    project = upsert_project(session, name="p", config_yaml="x")
    upsert_snapshot(
        session,
        project_id=project.id,
        fingerprint="fp",
        traces=_traces(4),
        default_split="dev",
    )
    rows = session.scalars(select(SplitAssignment)).all()
    assert len(rows) == 4
    assert {r.split for r in rows} == {"dev"}
    assert not any(r.sealed for r in rows)


def test_test_split_is_marked_sealed(session: Session) -> None:
    """Only the promotion gate may read sealed rows, so the flag has to be set
    at assignment time rather than inferred later."""
    project = upsert_project(session, name="p", config_yaml="x")
    upsert_snapshot(
        session,
        project_id=project.id,
        fingerprint="fp",
        traces=_traces(2),
        default_split="test",
    )
    assert all(r.sealed for r in session.scalars(select(SplitAssignment)).all())


def test_upsert_project_is_idempotent_and_refreshes_config(session: Session) -> None:
    first = upsert_project(session, name="p", config_yaml="version: 1")
    second = upsert_project(session, name="p", config_yaml="version: 2")
    assert first.id == second.id
    assert second.config_yaml == "version: 2"


def test_judge_config_is_keyed_by_hash(session: Session) -> None:
    """Re-registering an identical judge is a no-op; a changed rubric is a new
    row rather than an overwrite, so old results stay interpretable."""
    args = {
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "params": {"temperature": 0.0},
        "response_schema": {"type": "object"},
    }
    a = upsert_judge_config(session, version_hash="h1", **args)  # type: ignore[arg-type]
    b = upsert_judge_config(session, version_hash="h1", **args)  # type: ignore[arg-type]
    c = upsert_judge_config(session, version_hash="h2", **args)  # type: ignore[arg-type]
    assert a is b
    assert c.hash == "h2"


def test_record_results_rolls_up_cost_and_cache_hits(session: Session) -> None:
    """Aggregates are maintained on write so a run aborted by a budget limit can
    still say what it spent."""
    project = upsert_project(session, name="p", config_yaml="x")
    snapshot, _ = upsert_snapshot(
        session, project_id=project.id, fingerprint="fp", traces=_traces()
    )
    run = start_run(session, snapshot_id=snapshot.id, suite_hash="sh", split="dev")

    written = record_results(
        session,
        run_id=run.id,
        results=[
            EvalResult(
                trace_id="t0",
                evaluator_id="judge_q",
                evaluator_version="v1",
                passed=False,
                cost_usd=0.004,
                tokens_in=800,
                tokens_out=40,
            ),
            EvalResult(
                trace_id="t1",
                evaluator_id="judge_q",
                evaluator_version="v1",
                passed=True,
                cost_usd=0.0,
                tokens_in=0,
                tokens_out=0,
                cache_hit=True,
            ),
        ],
    )

    assert written == 2
    assert run.cost_usd == 0.004
    assert run.tokens_in == 800
    assert run.cache_hits == 1


def test_unpriced_result_does_not_corrupt_the_ledger(session: Session) -> None:
    """cost_usd None means "unknown price", not "free". It must not silently
    add zero and make an unpriced run look accounted for."""
    project = upsert_project(session, name="p", config_yaml="x")
    snapshot, _ = upsert_snapshot(
        session, project_id=project.id, fingerprint="fp", traces=_traces(1)
    )
    run = start_run(session, snapshot_id=snapshot.id, suite_hash="sh", split="dev")
    record_results(
        session,
        run_id=run.id,
        results=[EvalResult(trace_id="t0", evaluator_id="e", evaluator_version="v1", passed=True)],
    )
    stored = session.scalars(select(EvalResultRow)).one()
    assert stored.cost_usd is None
    assert run.cost_usd == 0.0


def test_scalar_prediction_is_wrapped_for_the_json_column(session: Session) -> None:
    """Predictions are arbitrary - a bool, a string, a list of tool calls.
    Wrapping keeps the column an object so readers handle one shape."""
    project = upsert_project(session, name="p", config_yaml="x")
    snapshot, _ = upsert_snapshot(
        session, project_id=project.id, fingerprint="fp", traces=_traces(1)
    )
    run = start_run(session, snapshot_id=snapshot.id, suite_hash="sh", split="dev")
    record_results(
        session,
        run_id=run.id,
        results=[
            EvalResult(
                trace_id="t0",
                evaluator_id="e",
                evaluator_version="v1",
                passed=True,
                normalized_prediction=True,
            )
        ],
    )
    assert session.scalars(select(EvalResultRow)).one().normalized_prediction == {"value": True}


def test_recording_results_for_an_unknown_run_raises(session: Session) -> None:
    """Results with no run are unattributable. Failing loudly beats writing rows
    nothing can ever query."""
    with pytest.raises(LookupError, match="no such run"):
        record_results(session, run_id="does-not-exist", results=[])


def test_dict_prediction_is_stored_unwrapped(session: Session) -> None:
    """A tool-call prediction is already an object and must not gain a spurious
    `value` nesting level."""
    project = upsert_project(session, name="p", config_yaml="x")
    snapshot, _ = upsert_snapshot(
        session, project_id=project.id, fingerprint="fp", traces=_traces(1)
    )
    run = start_run(session, snapshot_id=snapshot.id, suite_hash="sh", split="dev")
    prediction = {"name": "cancel_order", "arguments": {"order_id": "ORD-42"}}
    record_results(
        session,
        run_id=run.id,
        results=[
            EvalResult(
                trace_id="t0",
                evaluator_id="e",
                evaluator_version="v1",
                passed=False,
                normalized_prediction=prediction,
            )
        ],
    )
    assert session.scalars(select(EvalResultRow)).one().normalized_prediction == prediction
