"""Writing into nested structures - the inverse of resolve_path, and what lets a
mapping be written the way a user thinks about it."""

from __future__ import annotations

from typing import Any

import pytest

from evalloop.contracts import resolve_path, set_path


def test_creates_intermediate_dicts() -> None:
    target: dict[str, Any] = {}
    set_path(target, "input.user_request", "hello")
    assert target == {"input": {"user_request": "hello"}}


def test_creates_intermediate_lists_and_grows_them() -> None:
    """`output.artifacts[0].uri` is a reasonable thing to map into an empty
    trace, so the list has to come into being on demand."""
    target: dict[str, Any] = {}
    set_path(target, "output.artifacts[0].uri", "s3://b/a.wav")
    set_path(target, "output.artifacts[0].type", "audio")
    set_path(target, "output.artifacts[2].uri", "s3://b/c.wav")

    artifacts = target["output"]["artifacts"]
    assert artifacts[0] == {"uri": "s3://b/a.wav", "type": "audio"}
    assert artifacts[1] is None
    assert artifacts[2] == {"uri": "s3://b/c.wav"}


def test_round_trips_with_resolve_path() -> None:
    target: dict[str, Any] = {}
    for path, value in [
        ("trace_id", "t1"),
        ("output.tool_calls[0].name", "cancel_order"),
        ("output.tool_calls[0].arguments.order_id", "ORD-42"),
        ("metadata.language", "hi"),
    ]:
        set_path(target, path, value)
        assert resolve_path(target, path) == value


def test_existing_values_are_preserved() -> None:
    target: dict[str, Any] = {"input": {"user_request": "keep me"}}
    set_path(target, "input.system_prompt", "added")
    assert target["input"] == {"user_request": "keep me", "system_prompt": "added"}


def test_overwrites_a_leaf() -> None:
    target: dict[str, Any] = {}
    set_path(target, "output.text", "first")
    set_path(target, "output.text", "second")
    assert target["output"]["text"] == "second"


def test_none_is_writable_as_a_value() -> None:
    """`ground_truth.tool_calls: null` means "no tool should have been called" -
    a claim, not an absence - so it has to be storable."""
    target: dict[str, Any] = {}
    set_path(target, "ground_truth.tool_calls", None)
    assert target == {"ground_truth": {"tool_calls": None}}


def test_path_cannot_start_with_an_index() -> None:
    """The root of a trace is a mapping. `0.uri` would be indexing it.

    The bracket form `[0].uri` is rejected earlier still, by the path grammar
    itself - a bare index is not a valid segment."""
    with pytest.raises(ValueError, match="cannot start with a list index"):
        set_path({}, "0.uri", "x")

    with pytest.raises(ValueError, match="not a key, an index"):
        set_path({}, "[0].uri", "x")


def test_negative_index_cannot_be_written() -> None:
    """Reading `[-1]` is well defined; writing it is not - there is no sensible
    length for a list that does not exist yet."""
    with pytest.raises(ValueError, match="negative list indices"):
        set_path({}, "artifacts[-1].uri", "x")


def test_indexing_into_a_dict_is_a_type_error() -> None:
    target: dict[str, Any] = {"output": {"text": "hi"}}
    with pytest.raises(TypeError, match="cannot index into"):
        set_path(target, "output[0]", "x")


def test_keying_into_a_list_is_a_type_error() -> None:
    target: dict[str, Any] = {"items": ["a"]}
    with pytest.raises(TypeError, match="cannot set key"):
        set_path(target, "items.name", "x")


def test_writing_through_a_scalar_is_a_type_error() -> None:
    target: dict[str, Any] = {"output": {"text": "hi"}}
    with pytest.raises(TypeError):
        set_path(target, "output.text.deeper", "x")


def test_malformed_path_raises() -> None:
    with pytest.raises(ValueError):
        set_path({}, "a..b", "x")


def test_intermediate_index_into_a_dict_is_a_type_error() -> None:
    """`output[0].uri` where `output` is already a mapping. Two mapping lines
    that disagree about a container's shape, which is a config bug."""
    target: dict[str, Any] = {"output": {"text": "hi"}}
    with pytest.raises(TypeError, match="cannot index into dict"):
        set_path(target, "output[0].uri", "x")


def test_intermediate_key_into_a_list_is_a_type_error() -> None:
    target: dict[str, Any] = {"artifacts": [{"uri": "a"}]}
    with pytest.raises(TypeError, match="cannot set key"):
        set_path(target, "artifacts.first.uri", "x")


def test_writing_into_an_existing_list_element() -> None:
    """Two mapping lines targeting the same element must accumulate, not clobber
    - `artifacts[0].uri` and `artifacts[0].type` are one artifact."""
    target: dict[str, Any] = {}
    set_path(target, "output.artifacts[0].uri", "s3://b/a.wav")
    set_path(target, "output.artifacts[0].type", "audio")
    set_path(target, "output.artifacts[0].duration_ms", 1200)
    assert target["output"]["artifacts"] == [
        {"uri": "s3://b/a.wav", "type": "audio", "duration_ms": 1200}
    ]


def test_a_gap_left_by_a_skipped_index_stays_none() -> None:
    """Filling with None rather than {} means Trace validation reports the gap
    instead of silently accepting an empty artifact."""
    target: dict[str, Any] = {}
    set_path(target, "artifacts[1].uri", "s3://b/b.wav")
    assert target["artifacts"] == [None, {"uri": "s3://b/b.wav"}]


def test_final_index_into_an_existing_dict_is_a_type_error() -> None:
    """`output.0` where `output` was already built as a mapping by an earlier
    mapping line."""
    target: dict[str, Any] = {"output": {"text": "hi"}}
    with pytest.raises(TypeError, match="cannot index into dict"):
        set_path(target, "output.0", "x")


def test_a_whole_list_element_can_be_the_target() -> None:
    """`output.tool_calls[0]: first_call` maps an entire element rather than
    picking it apart field by field."""
    target: dict[str, Any] = {}
    set_path(target, "output.tool_calls[0]", {"name": "cancel_order", "arguments": {}})
    set_path(target, "output.tool_calls[2]", {"name": "issue_refund", "arguments": {}})

    calls = target["output"]["tool_calls"]
    assert calls[0]["name"] == "cancel_order"
    assert calls[1] is None
    assert calls[2]["name"] == "issue_refund"
