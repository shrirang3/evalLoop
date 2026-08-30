"""Path resolution is shared by mapping, evaluators, and the feedback compiler.
If it is wrong, it is wrong everywhere at once."""

from __future__ import annotations

import pytest

from evalloop.contracts import MISSING, path_exists, resolve_path, split_path

DATA = {
    "output": {
        "text": "hello",
        "tool_calls": [{"name": "cancel_order", "arguments": {"order_id": "ORD-42"}}],
        "artifacts": [{"uri": "s3://b/a.wav"}, {"uri": "s3://b/b.wav"}],
    },
    "ground_truth": {"tone": None},
}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("output.text", "hello"),
        ("output.tool_calls[0].name", "cancel_order"),
        ("output.tool_calls.0.name", "cancel_order"),
        ("output.artifacts[1].uri", "s3://b/b.wav"),
        ("output.artifacts[-1].uri", "s3://b/b.wav"),
        ("output.tool_calls[0].arguments.order_id", "ORD-42"),
    ],
)
def test_resolves(path: str, expected: str) -> None:
    assert resolve_path(DATA, path) == expected


@pytest.mark.parametrize(
    "path",
    ["output.missing", "output.artifacts[9].uri", "nope.at.all", "output.text.deeper"],
)
def test_absent_returns_missing_not_raises(path: str) -> None:
    assert resolve_path(DATA, path) is MISSING


def test_stored_none_is_present_but_falsy() -> None:
    """A recorded None is a real answer, not an absent key.

    The feedback compiler decides whether a target exists based on this
    distinction, so collapsing the two would silently fabricate targets."""
    assert resolve_path(DATA, "ground_truth.tone") is None
    assert path_exists(DATA, "ground_truth.tone")
    assert not path_exists(DATA, "ground_truth.absent")


@pytest.mark.parametrize("path", ["", "   ", "a..b", "a.b[", "a-b", "a.[0]"])
def test_malformed_path_raises(path: str) -> None:
    """A typo'd path is a config bug and must be loud, unlike absent data."""
    with pytest.raises(ValueError):
        split_path(path)


def test_missing_sentinel_is_falsy_and_not_none() -> None:
    assert not MISSING
    assert MISSING is not None


def test_index_into_non_sequence_returns_missing() -> None:
    """`output.text[0]` is a shape mismatch in the data, not a broken path.
    Report it as absent and let the caller say so, rather than crashing a run."""
    assert resolve_path(DATA, "output.text[0]") is MISSING
    assert resolve_path(DATA, "ground_truth[0]") is MISSING


def test_missing_repr_is_readable_in_errors() -> None:
    assert repr(MISSING) == "MISSING"
