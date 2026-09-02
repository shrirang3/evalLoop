"""Shared fixtures.

Unit tests build the schema from metadata on in-memory SQLite so they run with
no Docker and no network. Integration tests use the real Postgres, because
triggers, JSONB, and check constraints are exactly the things SQLite would let
through.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from evalloop.store.models import Base

PG_URL = os.environ.get(
    "EVALLOOP_TEST_DATABASE_URL",
    "postgresql+psycopg://evalloop:evalloop@localhost:5442/evalloop",
)


@pytest.fixture
def sqlite_engine() -> Iterator[Engine]:
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(sqlite_engine: Engine) -> Iterator[Session]:
    factory = sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    with factory() as s:
        yield s


@pytest.fixture(scope="session")
def pg_engine() -> Iterator[Engine]:
    """Real Postgres, or skip. Never silently degrade to SQLite - a green suite
    that skipped the constraint tests is worse than a red one."""
    engine = create_engine(PG_URL, future=True, pool_pre_ping=True)
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        engine.dispose()
        pytest.skip(f"metastore unreachable at {PG_URL}: {exc}")
    yield engine
    engine.dispose()


@pytest.fixture
def pg_session(pg_engine: Engine) -> Iterator[Session]:
    """A session whose writes are rolled back, so tests do not accumulate rows.

    Wrapped in an outer transaction rather than deleting afterwards, because
    snapshot and feedback_dataset rows cannot be deleted at all - that is the
    point of them.
    """
    connection = pg_engine.connect()
    transaction = connection.begin()
    factory = sessionmaker(bind=connection, expire_on_commit=False)
    session = factory()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def artifact_root(tmp_path: Path) -> Path:
    return tmp_path / "artifacts"


@pytest.fixture
def unique_name(request: pytest.FixtureRequest) -> str:
    """A per-test project name.

    Ingest commits for real, and `snapshot` rows cannot be deleted - that is the
    point of them - so tests cannot share a project without colliding on the
    unique name.
    """
    return f"t-{uuid.uuid4().hex[:12]}"
