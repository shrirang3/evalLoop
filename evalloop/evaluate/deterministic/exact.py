"""Exact string comparison, with the normalizations people actually want."""

from __future__ import annotations

import re
from typing import Any

from evalloop.contracts.paths import Missing
from evalloop.contracts.result import EvalResult
from evalloop.contracts.suite import EvaluatorSpec
from evalloop.contracts.trace import Trace
from evalloop.evaluate.base import not_applicable, resolve_or_missing, version_of

__all__ = ["ExactMatchEvaluator", "normalize"]

_WHITESPACE = re.compile(r"\s+")


def normalize(
    value: Any,
    *,
    lower: bool = False,
    strip: bool = True,
    collapse_whitespace: bool = False,
) -> str:
    text = "" if value is None else str(value)
    if strip:
        text = text.strip()
    if collapse_whitespace:
        text = _WHITESPACE.sub(" ", text)
    if lower:
        text = text.lower()
    return text


class ExactMatchEvaluator:
    """Compare two paths as strings."""

    def __init__(self, spec: EvaluatorSpec) -> None:
        self.spec = spec
        self.id = spec.id
        self._version = version_of(spec.version_payload())

        options = spec.options
        self._lower = bool(options.get("lower", False))
        self._strip = bool(options.get("strip", True))
        self._collapse = bool(options.get("collapse_whitespace", False))

    def version_hash(self) -> str:
        return self._version

    def evaluate(self, trace: Trace, ctx: Any) -> EvalResult:
        actual = resolve_or_missing(trace, self.spec.actual)
        expected = resolve_or_missing(trace, self.spec.expected)

        if isinstance(actual, Missing):
            return not_applicable(trace, self.id, self._version, f"no value at {self.spec.actual}")
        if self.spec.expected is None:
            return not_applicable(trace, self.id, self._version, "no 'expected' path configured")
        if isinstance(expected, Missing):
            return not_applicable(
                trace,
                self.id,
                self._version,
                f"no ground truth at {self.spec.expected}",
                prediction=normalize(
                    actual, lower=self._lower, strip=self._strip, collapse_whitespace=self._collapse
                ),
            )

        left = normalize(
            actual, lower=self._lower, strip=self._strip, collapse_whitespace=self._collapse
        )
        right = normalize(
            expected, lower=self._lower, strip=self._strip, collapse_whitespace=self._collapse
        )
        passed = left == right

        return EvalResult(
            trace_id=trace.trace_id,
            evaluator_id=self.id,
            evaluator_version=self._version,
            score=1.0 if passed else 0.0,
            passed=passed,
            normalized_prediction=left,
            ground_truth=right,
            explanation=None if passed else f"expected {right!r}, got {left!r}",
        )
