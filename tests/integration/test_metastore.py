"""Metastore constraints against real Postgres.

Everything here is something SQLite would happily allow: PL/pgSQL triggers,
check constraints, and JSONB. Running these on SQLite would produce a green
suite that proved nothing.
"""

from __future__ import annotations

import json

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError, ProgrammingError
from sqlalchemy.orm import Session

from evalloop.store import IMMUTABLE_TABLES, Project, Snapshot

pytestmark = pytest.mark.integration


def _project(session: Session, suffix: str = "") -> Project:
    project = Project(id=f"p{suffix}", name=f"proj{suffix}", config_yaml="name: x")
    session.add(project)
    session.flush()
    return project


def _snapshot(session: Session, project_id: str, suffix: str = "") -> Snapshot:
    snapshot = Snapshot(
        id=f"s{suffix}",
        project_id=project_id,
        source_fingerprint=f"fp{suffix}",
        row_count=10,
    )
    session.add(snapshot)
    session.flush()
    return snapshot


def test_migration_created_every_table(pg_engine: Engine) -> None:
    with pg_engine.connect() as conn:
        names = {
            row[0]
            for row in conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
            )
        }
    expected = {
        "project",
        "snapshot",
        "trace",
        "split_assignment",
        "eval_run",
        "eval_result",
        "judge_config",
        "llm_cache",
        "judgecard",
        "feedback_dataset",
        "train_run",
        "model_registry",
        "comparison",
    }
    assert expected <= names, f"missing: {expected - names}"


def test_immutability_triggers_exist(pg_engine: Engine) -> None:
    with pg_engine.connect() as conn:
        guarded = {
            row[0]
            for row in conn.execute(
                text(
                    "SELECT relname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
                    "WHERE NOT t.tgisinternal"
                )
            )
        }
    assert set(IMMUTABLE_TABLES) <= guarded


def test_snapshot_cannot_be_updated(pg_session: Session) -> None:
    """A judgecard from three months ago only means what it said if nothing
    could have edited the snapshot underneath it."""
    project = _project(pg_session, "u")
    snapshot = _snapshot(pg_session, project.id, "u")

    with pytest.raises((ProgrammingError, IntegrityError)) as exc:
        pg_session.execute(
            text("UPDATE snapshot SET row_count = 99 WHERE id = :i"), {"i": snapshot.id}
        )
    assert "immutable" in str(exc.value)


def test_snapshot_cannot_be_deleted(pg_session: Session) -> None:
    project = _project(pg_session, "d")
    snapshot = _snapshot(pg_session, project.id, "d")

    with pytest.raises((ProgrammingError, IntegrityError)) as exc:
        pg_session.execute(text("DELETE FROM snapshot WHERE id = :i"), {"i": snapshot.id})
    assert "immutable" in str(exc.value)


def test_mutable_tables_are_not_guarded(pg_session: Session) -> None:
    """Only evidence is frozen. A run has to be able to move from running to
    completed and accumulate cost as it goes."""
    project = _project(pg_session, "m")
    snapshot = _snapshot(pg_session, project.id, "m")
    pg_session.execute(
        text(
            "INSERT INTO eval_run (id, snapshot_id, suite_hash, split) "
            "VALUES ('rm', :s, 'sh', 'dev')"
        ),
        {"s": snapshot.id},
    )
    pg_session.execute(text("UPDATE eval_run SET status = 'completed' WHERE id = 'rm'"))
    status = pg_session.execute(text("SELECT status FROM eval_run WHERE id = 'rm'")).scalar_one()
    assert status == "completed"


def test_server_defaults_apply_to_raw_inserts(pg_session: Session) -> None:
    """Defaults declared only in Python never reach the database, so psql, raw
    SQL, and any other client would hit a NOT NULL with no default."""
    project = _project(pg_session, "sd")
    snapshot = _snapshot(pg_session, project.id, "sd")
    pg_session.execute(
        text(
            "INSERT INTO eval_run (id, snapshot_id, suite_hash, split) "
            "VALUES ('rsd', :s, 'sh', 'dev')"
        ),
        {"s": snapshot.id},
    )
    row = pg_session.execute(
        text("SELECT status, cost_usd, tokens_in, cache_hits FROM eval_run WHERE id = 'rsd'")
    ).one()
    assert row == ("running", 0.0, 0, 0)


def test_a_trace_cannot_be_in_two_splits(pg_session: Session) -> None:
    """The database-level half of rule 12: a trace that is both trained on and
    tested against invalidates the promotion decision."""
    project = _project(pg_session, "sp")
    snapshot = _snapshot(pg_session, project.id, "sp")
    insert = text(
        "INSERT INTO split_assignment (snapshot_id, trace_id, split, split_strategy) "
        "VALUES (:s, 't1', :split, 'hash_of_field')"
    )
    pg_session.execute(insert, {"s": snapshot.id, "split": "train"})
    with pytest.raises(IntegrityError):
        pg_session.execute(insert, {"s": snapshot.id, "split": "test"})


@pytest.mark.parametrize(
    ("sql", "constraint"),
    [
        (
            "INSERT INTO eval_run (id, snapshot_id, suite_hash, split) "
            "VALUES ('bad', :s, 'sh', 'validation')",
            "ck_eval_run_split",
        ),
        (
            "INSERT INTO trace (snapshot_id, trace_id, split, content_hash) "
            "VALUES (:s, 't', 'holdout', 'h')",
            "ck_trace_split",
        ),
        (
            "INSERT INTO comparison (id, baseline_model, candidate_model, run_ids, "
            "gate_result, decision) VALUES ('c', 'b', 'c', '{}', '{}', 'maybe')",
            "ck_comparison_decision",
        ),
    ],
)
def test_check_constraints_reject_invalid_enums(
    pg_session: Session, sql: str, constraint: str
) -> None:
    """Typed at the database rather than only in Pydantic, so a bad value cannot
    arrive through a migration, a fixture, or a hand-written INSERT."""
    project = _project(pg_session, "ck")
    snapshot = _snapshot(pg_session, project.id, "ck")
    with pytest.raises(IntegrityError) as exc:
        pg_session.execute(text(sql), {"s": snapshot.id})
    assert constraint in str(exc.value)


def test_jsonb_columns_round_trip_nested_structures(pg_session: Session) -> None:
    """Manifests and gate results are deeply nested. JSONB has to preserve them
    exactly, since the experiment bundle is meant to reproduce a decision."""
    manifest = {
        "target_source_histogram": {"judge_preference_pair": 412, "human_correction": 38},
        "dropped": {"no_target": 680},
        "judge_health": {"position_flip_rate": 0.04, "self_consistency": 0.91},
    }
    pg_session.execute(
        text(
            "INSERT INTO comparison (id, baseline_model, candidate_model, run_ids, "
            "gate_result, decision) VALUES ('cj', 'b', 'c', '{}', :g, 'reject')"
        ),
        {"g": json.dumps(manifest)},
    )
    stored = pg_session.execute(
        text("SELECT gate_result FROM comparison WHERE id = 'cj'")
    ).scalar_one()
    assert stored == manifest
