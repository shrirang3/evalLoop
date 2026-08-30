"""Project config and the cross-file integrity checks from plan/001 section 3."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalloop.contracts import JudgeConfig, ProjectConfig, check_integrity

PROJECT = {
    "name": "support-bot",
    "source": {"type": "jsonl", "path": "examples/support-bot/traces.jsonl"},
    "mapping": {"trace_id": "id", "input.user_request": "user_transcript"},
}


def test_minimal_project_parses_with_safe_defaults() -> None:
    project = ProjectConfig.model_validate(PROJECT)
    assert project.integrity.gate.deterministic_required is True
    assert project.integrity.gate.block_on_divergence is True
    assert project.splits.ratios["test"] == 0.15


def test_source_requires_the_field_its_type_needs() -> None:
    """A jsonl source with no path fails here rather than at ingest time, when
    someone has already waited for a connection."""
    with pytest.raises(ValidationError, match=r"requires source\.path"):
        ProjectConfig.model_validate({**PROJECT, "source": {"type": "jsonl"}})
    with pytest.raises(ValidationError, match=r"requires source\.query"):
        ProjectConfig.model_validate({**PROJECT, "source": {"type": "postgres"}})


def test_mapping_without_trace_id_rejected() -> None:
    """Without a stable id, results cannot be joined back to the trace that
    produced them."""
    with pytest.raises(ValidationError, match="trace_id"):
        ProjectConfig.model_validate({**PROJECT, "mapping": {"output.text": "reply"}})


def test_split_ratios_must_sum_to_one() -> None:
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        ProjectConfig.model_validate(
            {**PROJECT, "splits": {"ratios": {"train": 0.8, "dev": 0.3, "test": 0.1}}}
        )


def test_test_split_is_not_optional() -> None:
    """A project with no sealed test set cannot make a promotion claim at all."""
    with pytest.raises(ValidationError, match="sealed set is not optional"):
        ProjectConfig.model_validate({**PROJECT, "splits": {"ratios": {"train": 0.8, "dev": 0.2}}})


def test_unknown_key_is_an_error() -> None:
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate({**PROJECT, "redcation": []})


# --- integrity: base provider != judge provider (plan/001 section 3.1) ---

WITH_BASE = {
    **PROJECT,
    "models": {"base": {"provider": "anthropic", "model": "claude-sonnet-5"}},
}


def test_same_provider_for_base_and_judge_is_a_violation() -> None:
    """Judges measurably prefer their own family's outputs. Sharing a provider
    builds that bias into every number the tool reports."""
    judges = {"default": JudgeConfig(provider="anthropic", model="claude-opus-5")}
    violations = check_integrity(ProjectConfig.model_validate(WITH_BASE), judges)
    assert len(violations) == 1
    assert "require_distinct_providers" in violations[0]
    assert "anthropic" in violations[0]


def test_distinct_providers_pass() -> None:
    project = ProjectConfig.model_validate(
        {**PROJECT, "models": {"base": {"provider": "huggingface", "model": "Qwen/Qwen2.5-7B"}}}
    )
    judges = {"default": JudgeConfig(provider="anthropic", model="claude-sonnet-5")}
    assert check_integrity(project, judges) == []


def test_every_offending_judge_is_named() -> None:
    """Two judges on the same provider as the base model produce two
    violations, so the message names each rather than only the first."""
    judges = {
        "grader": JudgeConfig(provider="anthropic", model="claude-opus-5"),
        "tone": JudgeConfig(provider="anthropic", model="claude-sonnet-5"),
    }
    violations = check_integrity(ProjectConfig.model_validate(WITH_BASE), judges)
    assert len(violations) == 2
    assert {"'grader'" in v for v in violations} == {True, False}


def test_check_is_skipped_when_the_rule_is_disabled() -> None:
    """Opting out is allowed; doing it silently is not. The user must write it
    down in the config, where it is visible in review."""
    project = ProjectConfig.model_validate(
        {**WITH_BASE, "integrity": {"require_distinct_providers": []}}
    )
    judges = {"default": JudgeConfig(provider="anthropic", model="claude-opus-5")}
    assert check_integrity(project, judges) == []


def test_no_base_model_declared_means_nothing_to_check() -> None:
    """Evaluation-only projects never declare a base model and must not be
    blocked by a training-time rule."""
    judges = {"default": JudgeConfig(provider="anthropic", model="claude-sonnet-5")}
    assert check_integrity(ProjectConfig.model_validate(PROJECT), judges) == []


def test_empty_holdout_id_is_reported() -> None:
    project = ProjectConfig.model_validate(
        {**PROJECT, "integrity": {"gate": {"holdout_questions": ["  "]}}}
    )
    violations = check_integrity(project, {})
    assert any("holdout_questions" in v for v in violations)
