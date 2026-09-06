"""Is this call legal? - the tool check that needs no ground truth.

`json_match` asks "does this equal the recorded target?" and abstains on every
trace that has no target, which is every production trace anybody actually has
(`plan/002` "Why this document exists"). This asks a different question:

    is the tool real, permitted here, and correctly parameterised?

No target required, because legality is a property of the call and the registry,
not of a stored answer. That makes this the deterministic floor the promotion
gate needs (rule 10, `plan/001` section 3.2.1) on a customer's first run, with
zero labels - previously that floor existed only on paper, since the one
deterministic tool check silently returned `not_applicable` without ground truth.

What it deliberately does not catch: whether a *legal* call was the *right*
call. `issue_refund` on a 45-day-old order is registered, permitted, and
correctly typed. That judgement is `tool_selection`'s job.
"""

from __future__ import annotations

from typing import Any

from evalloop.contracts.paths import Missing
from evalloop.contracts.result import EvalResult
from evalloop.contracts.suite import EvaluatorSpec
from evalloop.contracts.tools import ToolRegistry
from evalloop.contracts.trace import Trace, canonical_json
from evalloop.evaluate.base import not_applicable, resolve_or_missing, version_of

__all__ = ["ADVISORY_CODES", "ToolRegistryCheckEvaluator"]

_ADVISORY = frozenset({"duplicate_undeclared"})
"""Recorded, reported, and not a failure.

A repeated call to a tool whose `side_effecting` the registry never declares is
unclassifiable, not wrong. Failing it would be a false failure - the expensive
direction (plan/001 section 5.3) - and passing it silently would hide the one
thing the author needs to fix.
"""

ADVISORY_CODES: frozenset[str] = _ADVISORY


class ToolRegistryCheckEvaluator:
    """Validate every tool call in a trace against the registry."""

    def __init__(self, spec: EvaluatorSpec, registry: ToolRegistry) -> None:
        if spec.expected is not None:
            raise ValueError(
                "tool_registry_check takes no 'expected' path - it validates a call "
                "against the registry rather than comparing it to a target. That is "
                "the point: it works on traces with no ground truth."
            )

        options = spec.options
        if options.get("check_preconditions"):
            raise ValueError(
                "check_preconditions is not implemented yet (plan/002 section 7). "
                "Declaring preconditions in tools.yaml is harmless - they are "
                "documentation until the expression engine lands - but switching "
                "this on would pass every trace while appearing to enforce them."
            )

        self.spec = spec
        self.id = spec.id
        self.registry = registry
        self.node_path: str | None = options.get("node_path")
        self.check_arguments = bool(options.get("check_arguments", True))
        self.check_duplicates = bool(options.get("check_duplicates", True))
        self._version = version_of({**spec.version_payload(), "registry": registry.registry_hash()})

    def version_hash(self) -> str:
        """Covers the registry as well as the spec.

        A tool description is prompt text for `tool_selection` and a contract
        for this check. Editing one without changing a version hash would let
        two incomparable runs compare clean (plan/002 rule 18).
        """
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
            # Calling nothing is legal, so this is not a failure - but counting
            # it as a pass would inflate the rate with traces the check never
            # looked at. Pass rate stays "of the traces that called something".
            return not_applicable(trace, self.id, self._version, "trace has no tool calls")

        violations = self._violations(trace, calls)
        failures = [v for v in violations if v["code"] not in _ADVISORY]
        passed = not failures
        reported = failures or violations

        return EvalResult(
            trace_id=trace.trace_id,
            evaluator_id=self.id,
            evaluator_version=self._version,
            score=1.0 if passed else 0.0,
            passed=passed,
            normalized_prediction=[_summarize(call) for call in calls],
            ground_truth=None,
            explanation="; ".join(v["message"] for v in reported) or None,
            # Structured as well as prose, so the wrong-tool report groups by
            # `code` instead of pattern-matching an English sentence that was
            # written for a human.
            raw_output={"violations": violations} if violations else None,
        )

    def _violations(self, trace: Trace, calls: Any) -> list[dict[str, Any]]:
        default_node = self._trace_node(trace)
        found: list[dict[str, Any]] = []
        seen: dict[str, int] = {}

        def record(code: str, index: int, tool: str | None, message: str) -> None:
            found.append(
                {"code": code, "call": index, "tool": tool, "message": f"call[{index}] {message}"}
            )

        for index, call in enumerate(calls):
            name = _attr(call, "name")
            if not isinstance(name, str):
                record("unnamed_call", index, None, "has no tool name")
                continue

            node = _attr(call, "node") or default_node
            arguments = _attr(call, "arguments") or {}

            if not self.registry.has(name):
                known = ", ".join(sorted(self.registry.tools))
                record(
                    "unregistered_tool",
                    index,
                    name,
                    f"'{name}' is not a registered tool; registered: {known}",
                )
                continue

            if not self.registry.knows_node(node):
                record(
                    "unknown_node",
                    index,
                    name,
                    f"'{name}' claims node '{node}', which is not in the registry",
                )
            elif name not in self.registry.allowed(node):
                allowed = ", ".join(sorted(self.registry.allowed(node)))
                record(
                    "not_permitted_at_node",
                    index,
                    name,
                    f"'{name}' is not permitted at node '{node}'; permitted here: {allowed}",
                )

            if self.check_arguments and isinstance(arguments, dict):
                for reason in self.registry.check_arguments(name, arguments):
                    record("invalid_arguments", index, name, f"'{name}': {reason}")

            if self.check_duplicates:
                fingerprint = canonical_json({"name": name, "arguments": arguments})
                first = seen.get(fingerprint)
                if first is None:
                    seen[fingerprint] = index
                elif self.registry.tools[name].side_effecting:
                    record(
                        "duplicate_side_effecting",
                        index,
                        name,
                        f"'{name}' repeats call[{first}] with identical arguments, "
                        f"and the tool is side-effecting",
                    )
                elif self.registry.tools[name].side_effecting is None:
                    record(
                        "duplicate_undeclared",
                        index,
                        name,
                        f"'{name}' repeats call[{first}], but the registry does not "
                        f"declare whether '{name}' is side-effecting",
                    )

        return found

    def _trace_node(self, trace: Trace) -> str | None:
        """A trace-level node, for products that record one per turn not per call."""
        if self.node_path is None:
            return None
        value = resolve_or_missing(trace, self.node_path)
        return value if isinstance(value, str) else None


def _attr(call: Any, name: str) -> Any:
    """Read a field from a ToolCall or from a plain dict.

    Traces reach evaluators as models, and tests and custom sources hand over
    dicts. Both are legitimate shapes for a call.
    """
    if isinstance(call, dict):
        return call.get(name)
    return getattr(call, name, None)


def _summarize(call: Any) -> dict[str, Any]:
    return {
        "name": _attr(call, "name"),
        "arguments": _attr(call, "arguments") or {},
        "node": _attr(call, "node"),
    }
