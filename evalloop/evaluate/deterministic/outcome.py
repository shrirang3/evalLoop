"""Did the call actually succeed?

The fourth tool check, and the second objective one. The other three ask about
the call as a *decision* - is it legal, was it the right tool, does the reply
match it - and all three pass on this trace:

    called:  issue_refund(order_id="ORD-8891", amount=79.99)
    result:  ERROR 409 - POLICY_VIOLATION, order is 45 days old
    said:    "Sure, I've processed your full refund."

Legal call, defensible tool, reply consistent with the call it made, and the
customer was told about money that never moved.

This reads the outcome the product already recorded (`plan/003` section 2). It
does not execute anything: `plan/000` P2 reserves `tool_call_exec` for sandbox
replay against a mock, which is a different mechanism with different risks, and
one name for two things helps nobody.

**The trap this evaluator exists to avoid** is reading "no outcome recorded" as
success. Most products log the invocation and not the return, so a check that
defaults to pass would report a flawless success rate for every customer whose
data cannot answer the question at all. Silence is `not_applicable`, always.
"""

from __future__ import annotations

from typing import Any

from evalloop.contracts.paths import Missing, resolve_path
from evalloop.contracts.result import EvalResult
from evalloop.contracts.suite import EvaluatorSpec
from evalloop.contracts.trace import Trace
from evalloop.evaluate.base import not_applicable, resolve_or_missing, version_of

__all__ = ["ToolCallOutcomeEvaluator"]

_DEFAULT_ERROR_FIELD = "error"


class ToolCallOutcomeEvaluator:
    """Fail a trace when any recorded tool call reports an error."""

    def __init__(self, spec: EvaluatorSpec) -> None:
        if spec.expected is not None:
            raise ValueError(
                "tool_call_outcome takes no 'expected' path - it reads the outcome the "
                "product recorded rather than comparing against a target"
            )

        options = spec.options
        self.spec = spec
        self.id = spec.id

        self.error_field: str = str(options.get("error_field", _DEFAULT_ERROR_FIELD))
        """Where the failure lives inside one call. A dotted path, so a product
        that encodes status in its return - `result.status` - is readable
        without a mapping step that cannot reach inside a list."""

        self.failure_values: list[Any] = list(options.get("failure_values", []))
        """When set, only these values count as failure, which is what makes
        `result.status` usable: `["error", "failed"]` rather than treating the
        string `"ok"` as a fault because it is merely present."""

        self._version = version_of(spec.version_payload())

    def version_hash(self) -> str:
        return self._version

    def evaluate(self, trace: Trace, ctx: Any) -> EvalResult:
        calls = resolve_or_missing(trace, self.spec.actual)
        if isinstance(calls, Missing):
            return not_applicable(trace, self.id, self._version, f"no value at {self.spec.actual}")
        if not isinstance(calls, (list, tuple)):
            return not_applicable(
                trace, self.id, self._version, f"{self.spec.actual} is not a list of tool calls"
            )
        if not calls:
            return not_applicable(trace, self.id, self._version, "trace has no tool calls")

        failures: list[dict[str, Any]] = []
        recorded = 0

        for index, call in enumerate(calls):
            outcome = _read(call, self.error_field)
            if isinstance(outcome, Missing) or outcome is None:
                continue
            recorded += 1
            if self._is_failure(outcome):
                failures.append(
                    {
                        "call": index,
                        "tool": _read(call, "name"),
                        "error": _short(outcome),
                    }
                )

        if recorded == 0:
            # The whole point. A product that logs invocations and not returns
            # cannot answer this question, and saying "pass" would answer it
            # anyway - with a flawless success rate that means nothing.
            return not_applicable(
                trace,
                self.id,
                self._version,
                f"no call records an outcome at '{self.error_field}'",
            )

        passed = not failures
        return EvalResult(
            trace_id=trace.trace_id,
            evaluator_id=self.id,
            evaluator_version=self._version,
            score=1.0 if passed else 0.0,
            passed=passed,
            normalized_prediction=[_summarize(call, self.error_field) for call in calls],
            # No target: the product's own verdict is the answer, not something
            # this check compares the answer against (plan/003 section 2).
            ground_truth=None,
            explanation="; ".join(
                f"call[{f['call']}] {f['tool']!r} failed: {f['error']}" for f in failures
            )
            or None,
            raw_output={"failures": failures, "recorded": recorded} if failures else None,
        )

    def _is_failure(self, outcome: Any) -> bool:
        if self.failure_values:
            return outcome in self.failure_values
        # No allowlist configured, so the field is a plain error slot: anything
        # present and non-empty is a failure.
        return bool(outcome) if not isinstance(outcome, str) else bool(outcome.strip())


def _read(call: Any, path: str) -> Any:
    """Read a dotted path inside one call, from a model or a plain dict."""
    if isinstance(call, dict):
        return resolve_path(call, path)
    return resolve_path(call, path)


def _short(value: Any, limit: int = 120) -> str:
    text = value if isinstance(value, str) else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _summarize(call: Any, error_field: str) -> dict[str, Any]:
    outcome = _read(call, error_field)
    return {
        "name": _read(call, "name"),
        "outcome": None if isinstance(outcome, Missing) else _short(outcome) if outcome else None,
    }
