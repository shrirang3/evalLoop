"""Direct tests for the annotation walk and node descent.

Both encode type-system trickiness that is hard to reach through YAML alone -
optional spellings, discriminated tags, scalar nodes. Testing them here keeps
the defensive branches honest instead of merely unreached.
"""

from __future__ import annotations

from typing import Optional, Union

import yaml
from pydantic import BaseModel

from evalloop.config.loader import (
    _first_model,
    _hint_for,
    _member_for_tag,
    _model_at,
    _strip_annotated,
    _unwrap,
)
from evalloop.config.yaml_positions import _descend
from evalloop.contracts import EvaluatorSpec, LLMQuestionSpec, ProjectConfig, SuiteEvaluator


class Leaf(BaseModel):
    value: int = 0


class Holder(BaseModel):
    leaf: Leaf = Leaf()
    optional_leaf: Leaf | None = None
    legacy_optional: Optional[Leaf] = None  # noqa: UP045 - both spellings on purpose
    leaves: list[Leaf] = []
    by_name: dict[str, Leaf] = {}


# --- _hint_for early exits ---


def test_no_hint_for_a_non_extra_error() -> None:
    """A type error already says what is wrong; a suggestion would be noise."""
    assert _hint_for({"type": "missing", "loc": ("x",), "msg": "", "input": None}, Leaf, ()) is None


def test_no_hint_without_a_location() -> None:
    error = {"type": "extra_forbidden", "loc": (), "msg": "", "input": None}
    assert _hint_for(error, Leaf, ()) is None  # type: ignore[arg-type]


def test_no_hint_when_the_owning_model_cannot_be_resolved() -> None:
    error = {"type": "extra_forbidden", "loc": ("nope", "deeper"), "msg": "", "input": None}
    assert _hint_for(error, Leaf, ()) is None  # type: ignore[arg-type]


# --- _model_at ---


def test_resolves_through_both_optional_spellings() -> None:
    """`X | None` yields types.UnionType and `Optional[X]` yields typing.Union.
    They are different objects, and checking only one silently drops every hint
    under an optional field."""
    assert _model_at(Holder, ("optional_leaf",)) is Leaf
    assert _model_at(Holder, ("legacy_optional",)) is Leaf


def test_resolves_through_a_list_and_a_dict() -> None:
    assert _model_at(Holder, ("leaves", 0)) is Leaf
    assert _model_at(Holder, ("by_name", "anything")) is Leaf


def test_unknown_field_yields_nothing() -> None:
    assert _model_at(Holder, ("absent",)) is None


def test_scalar_field_yields_nothing() -> None:
    assert _model_at(Leaf, ("value",)) is None


def test_an_unfollowable_path_offers_nothing() -> None:
    """Naming the wrong model's keys is worse than staying quiet: it reads as
    authoritative and sends the reader looking in the wrong place."""
    assert _model_at(Holder, ("leaf", "not_a_field", "deeper")) is None


def test_resolves_a_real_discriminated_union_by_tag() -> None:
    assert _model_at(ProjectConfig, ("models", "base")) is not None


# --- _member_for_tag ---


def test_member_for_tag_picks_the_named_member() -> None:
    assert _member_for_tag(SuiteEvaluator, "llm_question") is LLMQuestionSpec
    assert _member_for_tag(SuiteEvaluator, "json_match") is EvaluatorSpec


def test_member_for_tag_rejects_an_unknown_tag() -> None:
    assert _member_for_tag(SuiteEvaluator, "fuzzy_match") is None


def test_member_for_tag_ignores_non_string_segments() -> None:
    assert _member_for_tag(SuiteEvaluator, 0) is None


def test_member_for_tag_on_a_union_without_a_type_field() -> None:
    assert _member_for_tag(Union[Leaf, Holder], "leaf") is None  # noqa: UP007


# --- _unwrap and helpers ---


def test_unwrap_returns_none_for_a_plain_scalar() -> None:
    assert _unwrap(int, "anything") is None


def test_unwrap_list_requires_an_integer_segment() -> None:
    assert _unwrap(list[Leaf], "not-an-index") is None
    assert _unwrap(list[Leaf], 0) is Leaf


def test_unwrap_bare_containers_have_no_element_type() -> None:
    assert _unwrap(list, 0) is None
    assert _unwrap(dict, "k") is None


def test_unwrap_optional_list_by_index() -> None:
    assert _unwrap(list[Leaf] | None, 0) is Leaf


def test_unwrap_union_of_scalars_resolves_to_nothing() -> None:
    assert _unwrap(int | str, "x") is None


def test_strip_annotated_passes_plain_types_through() -> None:
    assert _strip_annotated(Leaf) is Leaf


def test_first_model_finds_a_member_and_tolerates_none() -> None:
    assert _first_model(Leaf | None) is Leaf
    assert _first_model(int) is None


# --- _descend ---


def _nodes(text: str) -> yaml.Node:
    node = yaml.compose(text)
    assert node is not None
    return node


def test_descend_into_a_sequence_needs_an_integer() -> None:
    root = _nodes("items:\n  - a\n  - b\n")
    sequence = _descend(root, "items")
    assert sequence is not None
    assert _descend(sequence[1], "not-an-index") is None
    assert _descend(sequence[1], 1) is not None


def test_descend_off_the_end_of_a_sequence() -> None:
    root = _nodes("items:\n  - a\n")
    sequence = _descend(root, "items")
    assert sequence is not None
    assert _descend(sequence[1], 5) is None
    assert _descend(sequence[1], -5) is None


def test_descend_into_a_scalar_goes_nowhere() -> None:
    """A path that runs past a leaf is a data shape mismatch, reported as absent
    rather than raised - a run of ten thousand traces must not die on one."""
    root = _nodes("name: value\n")
    scalar = _descend(root, "name")
    assert scalar is not None
    assert _descend(scalar[1], "deeper") is None


class Empty(BaseModel):
    pass


def test_model_with_no_fields_offers_no_suggestion() -> None:
    """Nothing to suggest and nothing to list, so stay silent rather than
    printing an empty 'valid keys here:'."""
    error = {"type": "extra_forbidden", "loc": ("anything",), "msg": "", "input": None}
    assert _hint_for(error, Empty, ()) is None  # type: ignore[arg-type]


def test_path_that_fails_inside_a_container_offers_nothing() -> None:
    """A string index into a list cannot be followed, and guessing past it would
    name fields the reader has not reached."""
    assert _model_at(Holder, ("leaves", "not-an-index", "value")) is None


def test_unwrap_skips_the_none_member_before_giving_up() -> None:
    assert _unwrap(int | None, "x") is None


def test_unwrap_optional_model_by_field_name() -> None:
    assert _unwrap(Leaf | None, "value") is Leaf
