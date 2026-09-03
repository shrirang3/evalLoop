"""Structural comparison. This is the tool-call check.

`plan/000` calls tool-call matching the highest-value deterministic check, and
plan/001 section 3.2.1 makes it load-bearing: it is the signal in a promotion
gate that the training loop could not have optimised against, because a judge
cannot game a JSON comparison. If it is sloppy, the whole trusted-judge design
rests on nothing.

Sloppy here means false failures. Every option below exists because of a real
production mismatch that would otherwise be reported as the model getting
something wrong:

    ignore_order              two calls in a different order, same effect
    coerce_types              order_id 42 from the DB, "42" from the model
    treat_null_as_missing     `{"note": null}` and `{}` mean the same thing
    treat_empty_as_missing    `arguments: {}` and no arguments key likewise
    allow_extra_arguments     the model passed an optional argument as well
    ignore_paths              a request_id that is never going to match

`treat_empty_as_missing` earns its default the hard way. `ToolCall.arguments`
defaults to `{}`, so a model output always carries the key, while ground truth
written by hand routinely omits it - and without this, every such pair reported
a false failure for "unexpected arguments" on a call that matched perfectly.
"""

from __future__ import annotations

from typing import Any

from evalloop.contracts.paths import Missing, split_path
from evalloop.contracts.result import EvalResult
from evalloop.contracts.suite import EvaluatorSpec
from evalloop.contracts.trace import Trace, canonical_json
from evalloop.evaluate.base import not_applicable, resolve_or_missing, version_of

__all__ = ["JsonMatchEvaluator", "canonicalize", "compare"]


class JsonMatchEvaluator:
    """Compare two structures after normalizing away differences that do not matter."""

    def __init__(self, spec: EvaluatorSpec) -> None:
        self.spec = spec
        self.id = spec.id
        self._version = version_of(spec.version_payload())

        options = spec.options
        self.ignore_order = bool(options.get("ignore_order", False))
        self.coerce_types = bool(options.get("coerce_types", False))
        self.treat_null_as_missing = bool(options.get("treat_null_as_missing", True))
        self.treat_empty_as_missing = bool(options.get("treat_empty_as_missing", True))
        self.allow_extra_arguments = bool(options.get("allow_extra_arguments", False))
        self.ignore_paths: list[str] = list(options.get("ignore_paths", []))

        for path in self.ignore_paths:
            split_path(path)  # fail on a typo'd ignore path at construction

    def version_hash(self) -> str:
        return self._version

    def evaluate(self, trace: Trace, ctx: Any) -> EvalResult:
        actual = resolve_or_missing(trace, self.spec.actual)
        expected = resolve_or_missing(trace, self.spec.expected)

        if isinstance(actual, Missing):
            return not_applicable(trace, self.id, self._version, f"no value at {self.spec.actual}")
        if self.spec.expected is None:
            return not_applicable(trace, self.id, self._version, "no 'expected' path configured")

        left = self._canonical(actual)
        if isinstance(expected, Missing):
            # The common case, and the one that must not read as a failure: the
            # model did something, and we have no record of what it should have
            # done.
            return not_applicable(
                trace,
                self.id,
                self._version,
                f"no ground truth at {self.spec.expected}",
                prediction=_jsonable(left),
            )

        right = self._canonical(expected)
        passed, reason = compare(left, right, allow_extra=self.allow_extra_arguments, path="$")

        return EvalResult(
            trace_id=trace.trace_id,
            evaluator_id=self.id,
            evaluator_version=self._version,
            score=1.0 if passed else 0.0,
            passed=passed,
            normalized_prediction=_jsonable(left),
            ground_truth=_jsonable(right),
            explanation=None if passed else reason,
        )

    def _canonical(self, value: Any) -> Any:
        return canonicalize(
            value,
            ignore_order=self.ignore_order,
            coerce_types=self.coerce_types,
            treat_null_as_missing=self.treat_null_as_missing,
            treat_empty_as_missing=self.treat_empty_as_missing,
            ignore_paths=self.ignore_paths,
        )


def canonicalize(
    value: Any,
    *,
    ignore_order: bool = False,
    coerce_types: bool = False,
    treat_null_as_missing: bool = True,
    treat_empty_as_missing: bool = True,
    ignore_paths: list[str] | None = None,
    path: str = "",
) -> Any:
    """Reduce a structure to the form the comparison should see."""
    ignore = ignore_paths or []
    if path and any(path == ignored or path.startswith(f"{ignored}.") for ignored in ignore):
        return _IGNORED

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if treat_null_as_missing and item is None:
                continue
            reduced = canonicalize(
                item,
                ignore_order=ignore_order,
                coerce_types=coerce_types,
                treat_null_as_missing=treat_null_as_missing,
                treat_empty_as_missing=treat_empty_as_missing,
                ignore_paths=ignore,
                path=child,
            )
            if reduced is _IGNORED:
                continue
            if treat_empty_as_missing and reduced in ({}, []):
                continue
            result[str(key)] = reduced
        return result

    if isinstance(value, list):
        items = [
            canonicalize(
                item,
                ignore_order=ignore_order,
                coerce_types=coerce_types,
                treat_null_as_missing=treat_null_as_missing,
                treat_empty_as_missing=treat_empty_as_missing,
                ignore_paths=ignore,
                path=path,
            )
            for item in value
        ]
        items = [item for item in items if item is not _IGNORED]
        if ignore_order:
            # Sorted by canonical JSON rather than by value: the elements are
            # dicts, which are not orderable, and this is stable.
            items = sorted(items, key=canonical_json)
        return items

    if coerce_types and isinstance(value, (int, float)) and not isinstance(value, bool):
        # Numeric identifiers are the whole reason for this. A database
        # returning 42 and a model emitting "42" mean the same order.
        return str(value)
    if coerce_types and isinstance(value, str):
        return value
    return value


def compare(
    left: Any,
    right: Any,
    *,
    allow_extra: bool = False,
    path: str = "$",
) -> tuple[bool, str | None]:
    """Compare canonicalized structures, reporting where they first diverge.

    The location matters more than the verdict. "tool_calls[0].arguments.amount:
    expected 0, got 79.99" is actionable; "did not match" sends someone reading
    two blobs of JSON side by side.
    """
    if isinstance(right, dict):
        if not isinstance(left, dict):
            return False, f"{path}: expected an object, got {_name(left)}"
        for key, expected in right.items():
            if key not in left:
                return False, f"{path}.{key}: missing"
            ok, reason = compare(left[key], expected, allow_extra=allow_extra, path=f"{path}.{key}")
            if not ok:
                return False, reason
        if not allow_extra:
            extra = sorted(set(left) - set(right))
            if extra:
                return False, f"{path}: unexpected {', '.join(extra)}"
        return True, None

    if isinstance(right, list):
        if not isinstance(left, list):
            return False, f"{path}: expected a list, got {_name(left)}"
        if len(left) != len(right):
            return False, f"{path}: expected {len(right)} item(s), got {len(left)}"
        for index, (got, expected) in enumerate(zip(left, right, strict=True)):
            ok, reason = compare(got, expected, allow_extra=allow_extra, path=f"{path}[{index}]")
            if not ok:
                return False, reason
        return True, None

    if left != right:
        return False, f"{path}: expected {right!r}, got {left!r}"
    return True, None


class _Ignored:
    """Marks a value removed by `ignore_paths`, so it is dropped rather than compared."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "IGNORED"


_IGNORED = _Ignored()


def _name(value: Any) -> str:
    return "null" if value is None else type(value).__name__


def _jsonable(value: Any) -> dict[str, Any]:
    """Wrap for the JSON column, which is typed as an object."""
    return value if isinstance(value, dict) else {"value": value}
