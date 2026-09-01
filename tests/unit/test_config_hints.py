"""Suggestion resolution - the reason unknown keys are errors rather than
ignored. `stratergy` becomes a question with an answer instead of a setting that
silently did nothing.

These exercise the annotation walk, which has to cross nested models, lists,
dicts, optionals, and discriminated unions to find the model that owns a key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evalloop.config import ConfigKind, load_config

PROJECT = """name: support-bot
source:
  type: jsonl
  path: traces.jsonl
mapping:
  trace_id: id
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _hints(tmp_path: Path, name: str, text: str, kind: ConfigKind | None = None) -> list[str]:
    _, problems = load_config(_write(tmp_path, name, text), kind=kind)
    return [p.hint for p in problems if p.hint]


def test_hint_at_the_top_level_of_a_model(tmp_path: Path) -> None:
    assert "did you mean 'description'?" in _hints(
        tmp_path, "project.yaml", PROJECT + "descriptoin: oops\n"
    )


def test_hint_inside_a_nested_model(tmp_path: Path) -> None:
    """source is a model within project, so the walk has to descend one level."""
    text = PROJECT.replace("  path: traces.jsonl", "  pth: traces.jsonl")
    assert "did you mean 'path'?" in _hints(tmp_path, "project.yaml", text)


def test_hint_inside_a_deeply_nested_model(tmp_path: Path) -> None:
    """integrity -> gate is two levels down."""
    text = PROJECT + "integrity:\n  gate:\n    determinstic_required: true\n"
    assert "did you mean 'deterministic_required'?" in _hints(tmp_path, "project.yaml", text)


def test_hint_inside_a_dict_valued_field(tmp_path: Path) -> None:
    """models is dict[Literal['base'], BaseModelSpec], so the walk must step
    through the value type rather than the mapping."""
    text = PROJECT + "models:\n  base:\n    provider: hf\n    modle: qwen\n"
    assert "did you mean 'model'?" in _hints(tmp_path, "project.yaml", text)


def test_hint_inside_a_list_of_models(tmp_path: Path) -> None:
    text = PROJECT + "redaction:\n  - type: regex\n    patern: '\\\\d+'\n"
    assert "did you mean 'pattern'?" in _hints(tmp_path, "project.yaml", text)


def test_hint_resolves_through_a_discriminated_union_tag(tmp_path: Path) -> None:
    """The tag names which member failed; resolving it is what lets the hint
    offer the right field instead of giving up."""
    text = (
        "suite: s\nevaluators:\n  - id: a\n    type: llm_question\n"
        "    question: q\n    holdut: true\n"
    )
    assert "did you mean 'holdout'?" in _hints(tmp_path, "eval-suite.yaml", text)


def test_unrecognisable_key_lists_valid_ones_instead_of_guessing(tmp_path: Path) -> None:
    """Below the similarity cutoff there is no sensible suggestion, so naming
    the alternatives beats an irrelevant one."""
    hints = _hints(tmp_path, "project.yaml", PROJECT + "zzzzzz: 1\n")
    assert len(hints) == 1
    assert hints[0].startswith("valid keys here:")
    assert "source" in hints[0]


def test_hints_only_apply_to_unknown_keys(tmp_path: Path) -> None:
    """A type error already says what is wrong; a suggestion would be noise."""
    text = PROJECT.replace("  type: jsonl", "  type: parquet")
    _, problems = load_config(_write(tmp_path, "project.yaml", text))
    assert problems
    assert all(p.hint is None for p in problems)


def test_promotion_gate_condition_hint(tmp_path: Path) -> None:
    """A list of models reached through a plain (non-discriminated) field."""
    text = "all:\n  - metric: tool_call_match\n    baselin_delta: 0.02\n"
    assert "did you mean 'baseline_delta'?" in _hints(tmp_path, "promotion.yaml", text)


def test_training_lora_hint(tmp_path: Path) -> None:
    text = "dataset_id: ds-1\nbase_model: qwen\nlora:\n  rnk: 8\n"
    assert "did you mean 'rank'?" in _hints(tmp_path, "training.yaml", text)


@pytest.mark.parametrize(
    "text",
    [
        # A key under a free-form dict cannot be wrong, so nothing is suggested.
        PROJECT + "splits:\n  ratios:\n    train: 0.7\n    dev: 0.15\n    test: 0.15\n",
        # Unknown key nested under an already-invalid parent: the walk stops and
        # simply offers no hint rather than crashing.
        PROJECT.replace("  type: jsonl", "  type: 12345") + "\n",
    ],
)
def test_unresolvable_paths_degrade_to_no_hint(tmp_path: Path, text: str) -> None:
    load_config(_write(tmp_path, "project.yaml", text))
