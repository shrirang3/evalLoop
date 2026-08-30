"""The normalized result stream.

Deterministic matchers and LLM judges produce very different things - a boolean
from a JSON comparison, a rubric answer from a model - but everything downstream
(judgecard, feedback compiler, promotion gate) reads one shape. This is that
shape.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["EvalResult", "JudgeResponse", "TokenUsage"]

_STRICT = ConfigDict(extra="forbid", frozen=True)


class TokenUsage(BaseModel):
    """Token counts for one judge call. Cost is first-class, not an afterthought."""

    model_config = _STRICT

    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = None
    """None means the model's price is unknown - deliberately not zero, so an
    unpriced model shows up as a gap in the ledger rather than as free."""


class JudgeResponse(BaseModel):
    """One round trip to a judge: what came back, what we made of it, what it cost."""

    model_config = _STRICT

    raw: str
    """The provider's response text, verbatim, before any parsing. Kept so a
    parse failure can be diagnosed without re-running the call."""

    parsed: dict[str, Any] | None = None
    """None when the response could not be coerced into the requested schema."""

    usage: TokenUsage = Field(default_factory=TokenUsage)
    judge_config_hash: str
    cache_hit: bool = False
    latency_ms: int | None = None
    error: str | None = None

    @property
    def invalid_output(self) -> bool:
        """True when the judge answered but the answer was unusable.

        Tracked separately from `error` (a transport or timeout failure). A judge
        that reliably returns malformed JSON is a broken judge, and the judgecard
        reports that rate as one of its headline numbers.
        """
        return self.parsed is None and self.error is None


class EvalResult(BaseModel):
    """One (trace, evaluator) outcome.

    `score` and `passed` are both optional because not every evaluator produces
    both: a set-comparison yields an F1 with no natural pass/fail, while an exact
    match yields a boolean with no meaningful score. Consumers must handle None
    rather than assume 0.0.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str
    evaluator_id: str
    evaluator_version: str
    """Config-derived hash of the evaluator. Stored on every row so results from
    two different versions of the same check are never silently compared."""

    score: float | None = None
    passed: bool | None = None

    normalized_prediction: Any = None
    """What the evaluator extracted from the trace, after its own normalization."""

    ground_truth: Any = None
    """What it was compared against. None when the check needed no ground truth."""

    explanation: str | None = None
    raw_output: dict[str, Any] | None = None

    # Judge provenance. Absent for deterministic evaluators.
    judge_config_hash: str | None = None
    cache_hit: bool = False
    invalid_output: bool = False

    error: str | None = None
    latency_ms: int | None = None
    cost_usd: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None

    run_id: str | None = None
    """Assigned when the row is written to the metastore, not by the evaluator."""

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_failure(self) -> bool:
        """True only for a check that ran and said no.

        An errored or invalid result is not a failure of the *model* - it is a
        failure of the evaluation - and must never be compiled into training data
        as though the model got something wrong.
        """
        return self.passed is False and self.error is None and not self.invalid_output
