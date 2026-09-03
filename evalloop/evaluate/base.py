"""Shared evaluator machinery.

Two things every evaluator needs and neither should reimplement: a config-derived
version hash, and the not-applicable result.

That second one carries a decision. `tool_call_match` needs
`ground_truth.tool_calls`, and plenty of real traces have no ground truth at
all. Calling that a failure invents a verdict the data does not support;
skipping the trace hides how little of the dataset the check can see. So the
row is written with `passed=None` and an explanation, which makes "how much of
my data can this check even reach?" a query rather than a guess - and
`EvalResult.is_failure` already returns False for it, so P4 can never compile
it into training data.
"""

from __future__ import annotations

import hashlib
from typing import Any

from evalloop.contracts.paths import MISSING, resolve_path
from evalloop.contracts.result import EvalResult
from evalloop.contracts.trace import Trace, canonical_json

__all__ = ["NOT_APPLICABLE", "error_result", "not_applicable", "resolve_or_missing", "version_of"]

NOT_APPLICABLE = "not applicable: {reason}"


def version_of(payload: dict[str, Any]) -> str:
    """Config-derived hash. Never runtime state, so the same YAML always agrees."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def resolve_or_missing(trace: Trace, path: str | None) -> Any:
    """Read a path, returning MISSING for both an absent path and an absent one."""
    if path is None:
        return MISSING
    return resolve_path(trace, path)


def not_applicable(
    trace: Trace,
    evaluator_id: str,
    version: str,
    reason: str,
    *,
    prediction: Any = None,
) -> EvalResult:
    """The check ran and had nothing to compare against."""
    return EvalResult(
        trace_id=trace.trace_id,
        evaluator_id=evaluator_id,
        evaluator_version=version,
        score=None,
        passed=None,
        normalized_prediction=prediction,
        ground_truth=None,
        explanation=NOT_APPLICABLE.format(reason=reason),
    )


def error_result(
    trace: Trace,
    evaluator_id: str,
    version: str,
    exc: Exception,
) -> EvalResult:
    """The evaluation itself broke.

    Distinct from a failure on purpose: a model that got the answer wrong and an
    evaluator that threw are different problems, and the second must never be
    compiled into training data as though it were the first.
    """
    return EvalResult(
        trace_id=trace.trace_id,
        evaluator_id=evaluator_id,
        evaluator_version=version,
        error=f"{type(exc).__name__}: {exc}",
    )
