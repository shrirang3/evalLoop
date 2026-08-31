"""Engine and session construction.

One place that knows how to reach the metastore, so tests can point at a
throwaway database and nothing else has to care.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

__all__ = ["DEFAULT_DATABASE_URL", "database_url", "make_engine", "session_scope"]

DEFAULT_DATABASE_URL = "postgresql+psycopg://evalloop:evalloop@localhost:5442/evalloop"


def database_url() -> str:
    """Metastore URL from the environment, falling back to the compose default."""
    return os.environ.get("EVALLOOP_DATABASE_URL", DEFAULT_DATABASE_URL)


def make_engine(url: str | None = None, *, echo: bool = False) -> Engine:
    return create_engine(url or database_url(), echo=echo, future=True, pool_pre_ping=True)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Commit on success, roll back on any exception.

    A half-written run is worse than no run: partial results still get read by
    the judgecard and quietly skew every aggregate computed from them.
    """
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
