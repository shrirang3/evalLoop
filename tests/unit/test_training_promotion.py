"""Training and promotion config. Shapes only at P0 - the engines are P5 and P6 -
but validating them now means a broken file fails before a GPU is booked."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalloop.contracts import PromotionConfig, TrainingConfig

TRAINING = {"dataset_id": "ds-1", "base_model": "Qwen/Qwen2.5-7B-Instruct"}


def test_training_defaults_refuse_tiny_and_truncated_runs() -> None:
    """Both guards exist because the failure is silent: a run on 12 pairs, or on
    targets cut off mid-sentence, completes happily and yields a candidate that
    looks trained and is not."""
    config = TrainingConfig.model_validate(TRAINING)
    assert config.min_rows == 50
    assert config.max_truncated_fraction == 0.05
    assert config.strategy == "dpo"


def test_negative_learning_rate_rejected() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate({**TRAINING, "learning_rate": -1e-4})


def test_unknown_training_key_rejected() -> None:
    with pytest.raises(ValidationError):
        TrainingConfig.model_validate({**TRAINING, "epocs": 3})


# --- promotion gate ---


def test_gate_condition_requires_exactly_one_comparison_target() -> None:
    """Structured rather than a parsed string, so an ambiguous condition fails
    at validate time instead of after training has been paid for."""
    with pytest.raises(ValidationError, match="exactly one"):
        PromotionConfig.model_validate({"all": [{"metric": "tool_call_match"}]})

    with pytest.raises(ValidationError, match="exactly one"):
        PromotionConfig.model_validate(
            {"all": [{"metric": "tool_call_match", "baseline_delta": 0.02, "absolute": 0.9}]}
        )


def test_valid_gate_parses() -> None:
    gate = PromotionConfig.model_validate(
        {
            "all": [
                {"metric": "tool_call_match", "baseline_delta": 0.02, "significant": True},
                {"metric": "invalid_output_rate", "op": "<=", "absolute": 0.02},
            ],
            "slices": [{"field": "metadata.language", "min_n": 25}],
        }
    )
    assert len(gate.all) == 2
    assert gate.slices[0].on_insufficient == "warn"


def test_empty_gate_rejected() -> None:
    """A gate with no conditions promotes everything, which is worse than having
    no gate at all because it looks like one."""
    with pytest.raises(ValidationError, match="promote everything"):
        PromotionConfig.model_validate({"slices": [{"field": "metadata.language"}]})


def test_insufficient_slice_data_never_silently_passes() -> None:
    slice_rule = PromotionConfig.model_validate(
        {"all": [{"metric": "x", "absolute": 0.5}], "slices": [{"field": "metadata.tier"}]}
    ).slices[0]
    assert slice_rule.on_insufficient in {"warn", "fail"}
    assert slice_rule.no_regression is True
