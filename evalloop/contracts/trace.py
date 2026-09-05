"""The common trace format.

Every product stores its data differently. Rather than ask customers to migrate,
EvalLoop asks them for a mapping, and everything downstream codes against this
one shape.

Two things about `ground_truth` are load-bearing, and both are decisions from
`plan/001-trusted-judge-architecture.md`:

1. It is **optional**. Most teams have production traces and no labels. A trace
   with no ground truth is completely valid and still evaluable.
2. It does two different jobs. `expected_response` / `tool_calls` are *targets*,
   consumed by the feedback compiler. `policy_followed` and friends are *labels*
   - a human verdict on a judge question - consumed by the judgecard. Both live
   in the same free-form dict because users invent their own key names.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from evalloop.contracts.paths import MISSING, resolve_path

__all__ = [
    "Artifact",
    "GroundTruth",
    "Message",
    "ToolCall",
    "Trace",
    "TraceInput",
    "TraceOutput",
]

ArtifactType = Literal["audio", "image", "file"]
Role = Literal["system", "user", "assistant", "tool"]

_STRICT = ConfigDict(extra="forbid", frozen=True)


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace.

    Used for every hash EvalLoop computes, so that two structurally identical
    objects always produce the same fingerprint regardless of key order.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


class Message(BaseModel):
    """One turn of conversation as the model saw it."""

    model_config = _STRICT

    role: Role
    content: str


class ToolCall(BaseModel):
    """A tool invocation, either the one that happened or the one that should have."""

    model_config = _STRICT

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    call_id: str | None = None

    node: str | None = None
    """Which node of the agent graph made this call.

    Optional, and unset is the common case: a flat single-node agent has no
    node to record. Where it is present, `tool_registry_check` can tell a scope
    violation (this tool exists, but not here) from a hallucination (this tool
    does not exist at all). Where it is absent the check degrades to global
    membership rather than failing (plan/002 section 1.1).
    """

    result: Any = None
    """What the tool returned, when the product records it.

    `plan/001` section 2 promises `tool_call_exec` at T0 - "did the call
    actually succeed" - and that is unanswerable from a call record alone. A
    structurally perfect call that returned POLICY_VIOLATION while the agent
    said "done" is the failure mode this field exists to make visible.
    """

    error: str | None = None
    """The tool's error, when it failed. Distinct from `result` being empty:
    a call that returned nothing and a call that raised are different events."""


class Artifact(BaseModel):
    """A pointer to a large binary that lives outside the trace.

    URI only - EvalLoop never stores bytes inside a trace. Audio recordings are
    gigabytes in aggregate and belong in the artifact store or the customer's own
    bucket, not in Postgres and not in a JSONL row.
    """

    model_config = _STRICT

    type: ArtifactType
    uri: str
    duration_ms: int | None = None
    mime: str | None = None

    @field_validator("uri")
    @classmethod
    def _reject_inline_bytes(cls, v: str) -> str:
        if v.startswith("data:"):
            raise ValueError(
                "artifact uri must reference external storage, not embed the bytes; got a data: URI"
            )
        if not v.strip():
            raise ValueError("artifact uri is empty")
        return v


class TraceInput(BaseModel):
    """What the model was given."""

    model_config = _STRICT

    messages: list[Message] = Field(default_factory=list)
    user_request: str | None = None
    system_prompt: str | None = None
    tools_available: list[dict[str, Any]] | None = None


class TraceOutput(BaseModel):
    """What the model actually produced. For a failing trace, this is the wrong answer."""

    model_config = _STRICT

    text: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)


class GroundTruth(BaseModel):
    """Free-form correctness data, keyed however the user likes.

    Not a fixed schema on purpose: one team records `expected_response`, another
    `gold_reply`, another a nested `review.approved_tool_call`. EvalLoop reads it
    by path, so the user names their own keys in YAML and nothing here has to
    know about them in advance.
    """

    model_config = ConfigDict(extra="allow", frozen=True)

    def has(self, path: str) -> bool:
        """True if `path` resolves, including to a stored None.

        A recorded `None` is a real answer ("no tool should have been called")
        and is not the same as the key being absent.
        """
        return resolve_path(self.model_dump(), path) is not MISSING

    def get(self, path: str, default: Any = None) -> Any:
        value = resolve_path(self.model_dump(), path)
        return default if value is MISSING else value

    @property
    def is_empty(self) -> bool:
        """True when no ground truth was supplied at all - the common case."""
        return not self.model_dump()


class Trace(BaseModel):
    """One production interaction, normalized.

    `content_hash` is derived, never stored on the instance, so it cannot drift
    from the data it describes. It covers the trace's *content* - id, input,
    output, ground truth, metadata - and deliberately excludes ingestion
    bookkeeping (`source_id`, `ingested_at`), so re-ingesting the same row
    tomorrow produces the same hash it did today.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1)
    input: TraceInput
    output: TraceOutput
    ground_truth: GroundTruth = Field(default_factory=GroundTruth)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Ingestion bookkeeping, set by the connector rather than the customer.
    source_id: str | None = None
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def content_hash(self) -> str:
        payload = {
            "trace_id": self.trace_id,
            "input": self.input.model_dump(mode="json"),
            "output": self.output.model_dump(mode="json"),
            "ground_truth": self.ground_truth.model_dump(mode="json"),
            "metadata": self.metadata,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    def resolve(self, path: str) -> Any:
        """Read any field by dotted path, e.g. `output.tool_calls[0].name`.

        This is what evaluator and mapping YAML resolves against.
        """
        return resolve_path(self, path)
