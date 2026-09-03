"""The Postgres judge cache.

Rule 4. The point is not the money it saves: it is that a cache keyed by judge
version cannot serve an answer given to a different question.
"""

from __future__ import annotations

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import sessionmaker

from evalloop.judge.cache import PostgresCache
from evalloop.store.models import LLMCache

pytestmark = pytest.mark.integration

RESPONSE = {"raw": '{"answer": true}', "parsed": {"answer": True}}
USAGE = {"tokens_in": 120, "tokens_out": 8, "cost_usd": 0.002}


@pytest.fixture
def cache(pg_engine: Engine, unique_name: str) -> PostgresCache:
    return PostgresCache(pg_engine)


def test_a_miss_returns_none_and_is_counted(cache: PostgresCache, unique_name: str) -> None:
    assert cache.get(f"absent-{unique_name}") is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_a_stored_answer_round_trips(cache: PostgresCache, unique_name: str) -> None:
    cache.put(unique_name, judge_config_hash="jh1", response=RESPONSE, usage=USAGE)
    hit = cache.get(unique_name)

    assert hit is not None
    response, usage = hit
    assert response == RESPONSE
    assert usage == USAGE
    assert cache.hits == 1


def test_writing_the_same_key_twice_is_a_no_op(
    cache: PostgresCache, pg_engine: Engine, unique_name: str
) -> None:
    """Two workers evaluating the same trace must not collide on the primary
    key and take the run down."""
    cache.put(unique_name, judge_config_hash="jh1", response=RESPONSE, usage=USAGE)
    cache.put(unique_name, judge_config_hash="jh1", response={"raw": "different"}, usage=USAGE)

    with sessionmaker(bind=pg_engine)() as session:
        count = session.scalar(
            select(func.count()).select_from(LLMCache).where(LLMCache.key == unique_name)
        )
        row = session.get(LLMCache, unique_name)

    assert count == 1
    assert row is not None
    assert row.response == RESPONSE  # the first write stands


def test_the_judge_hash_is_recorded_alongside_the_answer(
    cache: PostgresCache, pg_engine: Engine, unique_name: str
) -> None:
    """So a rubric change can be traced to the cache entries it orphaned."""
    cache.put(unique_name, judge_config_hash="jh-abc", response=RESPONSE, usage=USAGE)
    with sessionmaker(bind=pg_engine)() as session:
        row = session.get(LLMCache, unique_name)
    assert row is not None
    assert row.judge_config_hash == "jh-abc"


def test_different_keys_do_not_collide(cache: PostgresCache, unique_name: str) -> None:
    cache.put(f"{unique_name}-a", judge_config_hash="jh1", response={"raw": "a"}, usage=USAGE)
    cache.put(f"{unique_name}-b", judge_config_hash="jh1", response={"raw": "b"}, usage=USAGE)

    first = cache.get(f"{unique_name}-a")
    second = cache.get(f"{unique_name}-b")
    assert first is not None and first[0]["raw"] == "a"
    assert second is not None and second[0]["raw"] == "b"
