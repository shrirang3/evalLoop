"""Promotion gate configuration.

Shape only at P0 - the gate evaluates in P6. Conditions are structured rather
than parsed from strings like "candidate >= baseline + 0.02": a typo in a string
DSL fails at gate time, after training has already been paid for, whereas a
structured condition fails at `evalloop validate`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = ["GateCondition", "PromotionConfig", "SliceRule"]

_STRICT = ConfigDict(extra="forbid", frozen=True)

BuiltinMetric = Literal["invalid_output_rate", "cost_per_trace", "p95_latency_ms"]


class GateCondition(BaseModel):
    """One requirement the candidate must satisfy.

    Exactly one comparison target: an improvement over baseline, an absolute
    floor or ceiling, or a multiple of baseline.
    """

    model_config = _STRICT

    metric: str = Field(min_length=1)
    """An evaluator id (including a rubric sub-question) or a built-in."""

    op: Literal[">=", "<="] = ">="

    baseline_delta: float | None = None
    absolute: float | None = None
    baseline_factor: float | None = None

    significant: bool = False
    """Require the difference to survive a paired significance test, not just
    point in the right direction. A 0.4pp gain on 200 traces is noise."""

    @model_validator(mode="after")
    def _exactly_one_target(self) -> GateCondition:
        targets = [self.baseline_delta, self.absolute, self.baseline_factor]
        if sum(t is not None for t in targets) != 1:
            raise ValueError(
                "a gate condition needs exactly one of baseline_delta, absolute, or baseline_factor"
            )
        return self


class SliceRule(BaseModel):
    """Per-segment regression check.

    A candidate that improves overall while quietly breaking one language or one
    customer tier has not improved. Slices below `min_n` are reported as
    INSUFFICIENT DATA and never silently pass.
    """

    model_config = _STRICT

    field: str = Field(min_length=1)
    no_regression: bool = True
    min_n: int = Field(default=25, gt=0)
    tolerance: float = Field(default=0.02, ge=0.0)
    on_insufficient: Literal["warn", "ignore", "fail"] = "warn"


class PromotionConfig(BaseModel):
    """One `promotion.yaml`."""

    model_config = _STRICT

    all: list[GateCondition] = Field(default_factory=list)
    any: list[GateCondition] = Field(default_factory=list)
    slices: list[SliceRule] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_at_least_one_condition(self) -> PromotionConfig:
        if not self.all and not self.any:
            raise ValueError(
                "a promotion gate with no conditions would promote everything; "
                "declare at least one condition under 'all' or 'any'"
            )
        return self
