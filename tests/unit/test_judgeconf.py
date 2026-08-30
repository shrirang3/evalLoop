"""Judge configuration and its version hash. Rule 2 of the non-negotiables."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalloop.contracts import JudgeConfig, judge_version_hash

CFG = JudgeConfig(provider="anthropic", model="claude-sonnet-5")
SCHEMA = {"type": "object", "properties": {"answer": {"type": "boolean"}}}


def _hash(**overrides: object) -> str:
    kwargs: dict[str, object] = {
        "system_prompt": "You grade support transcripts.",
        "questions": ["Did the agent follow the refund policy?"],
        "response_schema": SCHEMA,
    }
    kwargs.update(overrides)
    config = kwargs.pop("config", CFG)
    return judge_version_hash(config, **kwargs)  # type: ignore[arg-type]


def test_one_word_prompt_edit_changes_the_hash() -> None:
    """The acceptance test for rule 2. If this ever passes silently, results
    from two different rubrics can be compared as if they were the same
    measurement."""
    before = _hash()
    after = _hash(questions=["Did the agent follow the returns policy?"])
    assert before != after


def test_same_config_is_stable_across_calls() -> None:
    assert _hash() == _hash()


def test_model_change_changes_the_hash() -> None:
    other = JudgeConfig(provider="anthropic", model="claude-opus-5")
    assert _hash() != _hash(config=other)


def test_temperature_change_changes_the_hash() -> None:
    hotter = JudgeConfig(provider="anthropic", model="claude-sonnet-5", temperature=0.7)
    assert _hash() != _hash(config=hotter)


def test_response_schema_change_changes_the_hash() -> None:
    """A different schema is a different question, even with identical wording."""
    wider = {"type": "object", "properties": {"answer": {"type": "integer"}}}
    assert _hash() != _hash(response_schema=wider)


def test_transport_settings_do_not_fragment_the_cache() -> None:
    """Pointing at a different replica of the same model cannot change what the
    model says. Including these would split the cache for no reason."""
    rerouted = JudgeConfig(
        provider="anthropic",
        model="claude-sonnet-5",
        base_url="https://proxy.internal/v1",
        timeout_s=120.0,
        max_retries=5,
    )
    assert _hash() == _hash(config=rerouted)


def test_temperature_defaults_to_zero() -> None:
    """A judge is an instrument. Sampling noise in an instrument reappears later
    as unexplained variance in the judgecard."""
    assert CFG.temperature == 0.0


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValidationError):
        JudgeConfig.model_validate(
            {"provider": "anthropic", "model": "claude-sonnet-5", "temprature": 0.5}
        )


def test_unknown_provider_rejected() -> None:
    with pytest.raises(ValidationError):
        JudgeConfig.model_validate({"provider": "gemini", "model": "x"})
