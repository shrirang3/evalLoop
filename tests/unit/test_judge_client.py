"""The judge client. Everything here is about the boundary never raising, and
the three outcomes staying distinct."""

from __future__ import annotations

from typing import Any

import pytest

from evalloop.contracts import JudgeConfig, RenderedPrompt
from evalloop.judge import (
    JudgeClient,
    MockProvider,
    ProviderError,
    RateLimited,
    ServerError,
    Timeout,
    cache_key,
    make_provider,
)

CONFIG = JudgeConfig(provider="mock", model="stub-1", max_retries=2)
SCHEMA = {"type": "object", "properties": {"answer": {"type": "boolean"}}}
PROMPT = RenderedPrompt(system="be terse", messages=[{"role": "user", "content": "ok?"}])


def _client(provider: Any, **kwargs: Any) -> JudgeClient:
    return JudgeClient(
        CONFIG,
        provider,
        system_prompt="be terse",
        questions=["ok?"],
        response_schema=SCHEMA,
        sleep=lambda _: None,
        **kwargs,
    )


def test_a_usable_answer_is_parsed() -> None:
    response = _client(MockProvider(answers=[{"answer": True}])).ask(PROMPT, SCHEMA)
    assert response.parsed == {"answer": True}
    assert response.error is None
    assert not response.invalid_output


def test_transport_failure_sets_error_and_does_not_raise() -> None:
    """A run of ten thousand traces must not die because one call failed."""
    response = _client(MockProvider(raise_error=ProviderError("boom"))).ask(PROMPT, SCHEMA)
    assert response.error is not None
    assert "boom" in response.error
    assert response.parsed is None
    assert not response.invalid_output  # a failed call is not an unusable answer


def test_unusable_answer_is_invalid_output_not_an_error() -> None:
    """Distinct on purpose: a judge returning malformed JSON on a third of calls
    is broken, and a judge behind a flaky network is not. The fix differs."""
    provider = MockProvider(answers=["not json at all", "still not json"])
    response = _client(provider).ask(PROMPT, SCHEMA)
    assert response.invalid_output
    assert response.error is None
    assert response.parsed is None


def test_one_repair_attempt_is_made() -> None:
    """Models that wrap JSON in prose usually comply when told what went wrong."""
    provider = MockProvider(answers=["Sure! Here you go.", {"answer": False}])
    response = _client(provider).ask(PROMPT, SCHEMA)

    assert response.parsed == {"answer": False}
    assert len(provider.calls) == 2
    assert "could not be parsed" in provider.calls[1].messages[-1]["content"]


def test_repair_is_attempted_only_once() -> None:
    """A second failure is a property of the judge and belongs in the
    invalid-output rate, not in another round of retries."""
    provider = MockProvider(answers=["nope", "still nope", {"answer": True}])
    response = _client(provider).ask(PROMPT, SCHEMA)
    assert response.invalid_output
    assert len(provider.calls) == 2


def test_both_calls_are_billed_when_a_repair_succeeds() -> None:
    provider = MockProvider(answers=["prose", {"answer": True}], tokens_in=100, cost_usd=0.002)
    response = _client(provider).ask(PROMPT, SCHEMA)
    assert response.usage.tokens_in == 200
    assert response.usage.cost_usd == pytest.approx(0.004)


def test_unpriced_repair_keeps_the_cost_unknown() -> None:
    """None means unknown. Treating it as zero would make the repair look free."""
    provider = MockProvider(answers=["prose", {"answer": True}], cost_usd=None)
    assert _client(provider).ask(PROMPT, SCHEMA).usage.cost_usd is None


@pytest.mark.parametrize("error", [RateLimited("429"), ServerError("503")])
def test_rate_limits_and_server_errors_are_retried(error: Exception) -> None:
    provider = MockProvider(raise_error=error)
    response = _client(provider).ask(PROMPT, SCHEMA)
    assert len(provider.calls) == CONFIG.max_retries + 1
    assert response.error is not None


def test_timeouts_are_not_retried() -> None:
    """A judge that times out on a third of calls is a fact the judgecard
    reports. Retrying silently would hide it and triple the bill."""
    provider = MockProvider(raise_error=Timeout("timed out"))
    response = _client(provider).ask(PROMPT, SCHEMA)
    assert len(provider.calls) == 1
    assert "Timeout" in str(response.error)


def test_a_successful_retry_returns_the_answer() -> None:
    class FlakyOnce:
        name = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, prompt: Any, schema: Any, config: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                raise ServerError("503")
            return MockProvider(answers=[{"answer": True}]).complete(prompt, schema, config)

    provider = FlakyOnce()
    assert _client(provider).ask(PROMPT, SCHEMA).parsed == {"answer": True}
    assert provider.calls == 2


@pytest.mark.parametrize(
    "text",
    [
        '{"answer": true}',
        '```json\n{"answer": true}\n```',
        '```\n{"answer": true}\n```',
        'Here is my answer:\n{"answer": true}\nHope that helps.',
    ],
)
def test_json_is_recovered_from_fences_and_prose(text: str) -> None:
    """A judge that is right about the answer and wrong about the formatting is
    worth recovering. Guessing at the content is not."""
    assert _client(MockProvider(answers=[text])).ask(PROMPT, SCHEMA).parsed == {"answer": True}


@pytest.mark.parametrize("text", ["[1, 2, 3]", '"just a string"', "42", ""])
def test_non_object_replies_are_invalid(text: str) -> None:
    provider = MockProvider(answers=[text, text])
    assert _client(provider).ask(PROMPT, SCHEMA).invalid_output


# --- version hash and cache key ---


def test_version_hash_covers_the_question() -> None:
    """The same model asked a different question is a different instrument."""
    first = _client(MockProvider()).version_hash
    other = JudgeClient(
        CONFIG,
        MockProvider(),
        system_prompt="be terse",
        questions=["different question?"],
        response_schema=SCHEMA,
    )
    assert first != other.version_hash


def test_cache_key_changes_with_the_judge_version() -> None:
    """The point of the cache is not saving money; it is that a rubric edit can
    never reuse answers given to the old question."""
    assert cache_key("v1", PROMPT, SCHEMA) != cache_key("v2", PROMPT, SCHEMA)


def test_cache_key_changes_with_the_prompt() -> None:
    other = RenderedPrompt(system="be terse", messages=[{"role": "user", "content": "different"}])
    assert cache_key("v1", PROMPT, SCHEMA) != cache_key("v1", other, SCHEMA)


def test_cache_key_is_stable() -> None:
    assert cache_key("v1", PROMPT, SCHEMA) == cache_key("v1", PROMPT, SCHEMA)


# --- cache behaviour ---


class DictCache:
    def __init__(self) -> None:
        self.store: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        self.writes = 0

    def get(self, key: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        return self.store.get(key)

    def put(self, key: str, *, judge_config_hash: str, response: Any, usage: Any) -> None:
        self.store[key] = (response, usage)
        self.writes += 1


def test_a_second_identical_ask_hits_the_cache() -> None:
    provider = MockProvider(answers=[{"answer": True}])
    client = _client(provider, cache=DictCache())

    first = client.ask(PROMPT, SCHEMA)
    second = client.ask(PROMPT, SCHEMA)

    assert len(provider.calls) == 1
    assert not first.cache_hit
    assert second.cache_hit
    assert second.parsed == first.parsed


def test_a_cache_hit_costs_nothing() -> None:
    """Replaying the original usage would bill the same call twice and make a
    fully-cached run look as expensive as the first."""
    client = _client(MockProvider(answers=[{"answer": True}], cost_usd=0.002), cache=DictCache())
    client.ask(PROMPT, SCHEMA)
    hit = client.ask(PROMPT, SCHEMA)

    assert hit.usage.cost_usd == 0.0
    assert hit.usage.tokens_in == 0
    assert hit.usage.tokens_out == 0


def test_unusable_answers_are_not_cached() -> None:
    """Caching a malformed reply would make a transient formatting glitch
    permanent for that trace."""
    cache = DictCache()
    provider = MockProvider(answers=["nope", "still nope"])
    _client(provider, cache=cache).ask(PROMPT, SCHEMA)
    assert cache.writes == 0


def test_failed_calls_are_not_cached() -> None:
    cache = DictCache()
    _client(MockProvider(raise_error=Timeout("t")), cache=cache).ask(PROMPT, SCHEMA)
    assert cache.writes == 0


# --- provider factory ---


@pytest.mark.parametrize("name", ["anthropic", "openai_compat", "mock"])
def test_every_declared_provider_can_be_built(name: str) -> None:
    assert make_provider(name).name == name


def test_unknown_provider_names_the_available_ones() -> None:
    with pytest.raises(ValueError, match="unknown judge provider"):
        make_provider("gemini")


def test_a_non_retryable_error_during_repair_gives_up_cleanly() -> None:
    """The first reply was unusable and the repair call itself failed. That is
    invalid output, not an error - the judge did answer, just not usably."""

    class FailsOnRepair:
        name = "fails-on-repair"

        def __init__(self) -> None:
            self.calls: list[Any] = []

        def complete(self, prompt: Any, schema: Any, config: Any) -> Any:
            self.calls.append(prompt)
            if len(self.calls) == 1:
                return MockProvider(answers=["prose, not json"]).complete(prompt, schema, config)
            raise Timeout("repair timed out")

    provider = FailsOnRepair()
    response = _client(provider).ask(PROMPT, SCHEMA)

    assert response.invalid_output
    assert response.error is None
    assert len(provider.calls) == 2


def test_embedded_json_that_is_itself_malformed_stays_invalid() -> None:
    """Braces are found but the content between them is not JSON. Recovering a
    fence is worth doing; guessing at the content is not."""
    broken = 'Here you go: {"answer": tru}'
    assert _client(MockProvider(answers=[broken, broken])).ask(PROMPT, SCHEMA).invalid_output
