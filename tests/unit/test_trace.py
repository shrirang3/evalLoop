"""The trace contract. Everything downstream imports this shape, so a mistake
here propagates into every phase."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalloop.contracts import Trace

MINIMAL = {
    "trace_id": "sb-0417",
    "input": {"user_request": "I want a refund"},
    "output": {"text": "Sure, refunded."},
}


def test_ground_truth_is_optional() -> None:
    """The central decision of plan/001: most teams have traces and no labels.

    A trace without ground truth must be fully valid, not a degraded case."""
    trace = Trace.model_validate(MINIMAL)
    assert trace.ground_truth.is_empty
    assert not trace.ground_truth.has("expected_response")


def test_ground_truth_accepts_arbitrary_user_keys() -> None:
    """Users invent their own key names; nothing here knows them in advance."""
    trace = Trace.model_validate(
        {**MINIMAL, "ground_truth": {"gold_reply": "No refund after 30 days", "tier": 2}}
    )
    assert trace.ground_truth.has("gold_reply")
    assert trace.ground_truth.get("tier") == 2
    assert trace.ground_truth.get("absent", "fallback") == "fallback"


def test_ground_truth_distinguishes_stored_none_from_absent() -> None:
    """`tool_calls: None` means "no tool should have been called" - a real
    target. An absent key means we have no idea. The feedback compiler must
    never confuse the two."""
    trace = Trace.model_validate({**MINIMAL, "ground_truth": {"tool_calls": None}})
    assert trace.ground_truth.has("tool_calls")
    assert trace.ground_truth.get("tool_calls") is None
    assert not trace.ground_truth.has("expected_response")


def test_artifacts_are_uri_only() -> None:
    """Audio is gigabytes in aggregate. A trace carries a pointer, never bytes."""
    with pytest.raises(ValidationError, match="data: URI"):
        Trace.model_validate(
            {
                **MINIMAL,
                "output": {"artifacts": [{"type": "audio", "uri": "data:audio/wav;base64,UklGR"}]},
            }
        )


def test_unknown_field_is_an_error_not_a_warning() -> None:
    """Silently ignoring an unknown key is how a mapping typo survives to
    production and quietly drops a column."""
    with pytest.raises(ValidationError):
        Trace.model_validate({**MINIMAL, "trace_metadata": {"oops": 1}})


def test_content_hash_is_stable_under_key_reordering() -> None:
    a = Trace.model_validate({**MINIMAL, "metadata": {"language": "en", "tier": "premium"}})
    b = Trace.model_validate({**MINIMAL, "metadata": {"tier": "premium", "language": "en"}})
    assert a.content_hash == b.content_hash


def test_content_hash_ignores_ingestion_bookkeeping() -> None:
    """Re-ingesting the same source row tomorrow must produce the same hash,
    otherwise snapshot idempotency (P0.6) is impossible."""
    from datetime import UTC, datetime

    a = Trace.model_validate(MINIMAL)
    b = Trace.model_validate(
        {**MINIMAL, "source_id": "row-99", "ingested_at": datetime(2020, 1, 1, tzinfo=UTC)}
    )
    assert a.content_hash == b.content_hash


def test_content_hash_changes_with_content() -> None:
    a = Trace.model_validate(MINIMAL)
    b = Trace.model_validate({**MINIMAL, "output": {"text": "Sure, refunded!"}})
    assert a.content_hash != b.content_hash


def test_resolve_crosses_from_typed_fields_into_free_form_dicts() -> None:
    """One path syntax has to walk models, lists, and plain dicts alike, since
    evaluator YAML does not know which is which."""
    trace = Trace.model_validate(
        {
            **MINIMAL,
            "output": {
                "tool_calls": [{"name": "cancel_order", "arguments": {"order_id": "ORD-42"}}]
            },
            "metadata": {"language": "hi"},
        }
    )
    assert trace.resolve("output.tool_calls[0].name") == "cancel_order"
    assert trace.resolve("output.tool_calls[0].arguments.order_id") == "ORD-42"
    assert trace.resolve("metadata.language") == "hi"


def test_trace_is_immutable() -> None:
    """A trace is evidence. Nothing downstream may edit it in place."""
    trace = Trace.model_validate(MINIMAL)
    with pytest.raises(ValidationError):
        trace.trace_id = "different"  # type: ignore[misc]


def test_empty_trace_id_rejected() -> None:
    with pytest.raises(ValidationError):
        Trace.model_validate({**MINIMAL, "trace_id": ""})


def test_blank_artifact_uri_rejected() -> None:
    """An empty pointer is worse than no artifact - it looks like data until
    something tries to fetch it."""
    with pytest.raises(ValidationError, match="uri is empty"):
        Trace.model_validate(
            {**MINIMAL, "output": {"artifacts": [{"type": "audio", "uri": "   "}]}}
        )


def test_valid_artifact_reference_is_preserved() -> None:
    """The voice case: the recording stays in the customer's bucket and the
    trace carries only the pointer plus enough metadata to reason about it."""
    trace = Trace.model_validate(
        {
            **MINIMAL,
            "output": {
                "artifacts": [
                    {
                        "type": "audio",
                        "uri": "s3://bucket/call-123.wav",
                        "duration_ms": 45_000,
                        "mime": "audio/wav",
                    }
                ]
            },
        }
    )
    artifact = trace.output.artifacts[0]
    assert artifact.uri == "s3://bucket/call-123.wav"
    assert artifact.duration_ms == 45_000
    assert trace.resolve("output.artifacts[0].uri") == "s3://bucket/call-123.wav"
