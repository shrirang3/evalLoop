"""Engine and session construction."""

from __future__ import annotations

import pytest
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import sessionmaker

from evalloop.store import DEFAULT_DATABASE_URL, Project, database_url, make_engine, session_scope


def test_database_url_prefers_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALLOOP_DATABASE_URL", "postgresql+psycopg://u:p@host:1/db")
    assert database_url() == "postgresql+psycopg://u:p@host:1/db"


def test_database_url_falls_back_to_the_compose_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EVALLOOP_DATABASE_URL", raising=False)
    assert database_url() == DEFAULT_DATABASE_URL


def test_make_engine_accepts_an_explicit_url() -> None:
    engine = make_engine("sqlite://")
    try:
        with engine.connect() as conn:
            assert conn.execute(text("SELECT 1")).scalar_one() == 1
    finally:
        engine.dispose()


def test_session_scope_commits_on_success(sqlite_engine: Engine) -> None:
    with session_scope(sqlite_engine) as session:
        session.add(Project(id="p1", name="kept", config_yaml="x"))

    factory = sessionmaker(bind=sqlite_engine)
    with factory() as check:
        assert check.scalar(select(Project.name).where(Project.id == "p1")) == "kept"


def test_session_scope_rolls_back_on_error(sqlite_engine: Engine) -> None:
    """A half-written run is worse than no run: partial results still get read
    by the judgecard and quietly skew every aggregate built from them."""
    with pytest.raises(RuntimeError), session_scope(sqlite_engine) as session:
        session.add(Project(id="p2", name="discarded", config_yaml="x"))
        session.flush()
        raise RuntimeError("boom")

    factory = sessionmaker(bind=sqlite_engine)
    with factory() as check:
        assert check.scalar(select(Project).where(Project.id == "p2")) is None
