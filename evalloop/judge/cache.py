"""Judge responses in Postgres, keyed by judge version.

Rule 4. The point is not saving money, though it does: it is that a cache keyed
by judge version can never serve an answer given to a different question. A
rubric edit changes the hash, changes the key, and misses - so recalibrating a
prompt is safe by construction rather than by remembering to clear something.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Engine

from evalloop.store.db import session_scope
from evalloop.store.models import LLMCache

__all__ = ["PostgresCache"]


class PostgresCache:
    """Short sessions per operation, so nothing holds a connection across a run."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        with session_scope(self.engine) as session:
            row = session.get(LLMCache, key)
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
            return row.response, row.usage

    def put(
        self,
        key: str,
        *,
        judge_config_hash: str,
        response: dict[str, Any],
        usage: dict[str, Any],
    ) -> None:
        with session_scope(self.engine) as session:
            if session.get(LLMCache, key) is not None:
                return
            session.add(
                LLMCache(
                    key=key,
                    judge_config_hash=judge_config_hash,
                    response=response,
                    usage=usage,
                )
            )
