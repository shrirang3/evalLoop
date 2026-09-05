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

__all__ = ["ToolRegistryCheckEvaluator"]


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

        default_node = self._trace_node(trace)
        problems: list[str] = []
        notes: list[str] = []
        seen: dict[str, int] = {}

        for index, call in enumerate(calls):
            name = _attr(call, "name")
            if not isinstance(name, str):
                problems.append(f"call[{index}] has no tool name")
                continue

            node = _attr(call, "node") or default_node
            arguments = _attr(call, "arguments") or {}

            if not self.registry.has(name):
                known = ", ".join(sorted(self.registry.tools))
                problems.append(
                    f"call[{index}] '{name}' is not a registered tool; registered: {known}"
                )
                continue

            if not self.registry.knows_node(node):
                problems.append(
                    f"call[{index}] '{name}' claims node '{node}', which is not in the registry"
                )
            elif name not in self.registry.allowed(node):
                allowed = ", ".join(sorted(self.registry.allowed(node)))
                problems.append(
                    f"call[{index}] '{name}' is not permitted at node '{node}'; "
                    f"permitted here: {allowed}"
                )

            if self.check_arguments and isinstance(arguments, dict):
                problems.extend(
                    f"call[{index}] '{name}': {reason}"
                    for reason in self.registry.check_arguments(name, arguments)
                )

            if self.check_duplicates:
                fingerprint = canonical_json({"name": name, "arguments": arguments})
                first = seen.get(fingerprint)
                if first is not None:
                    side_effecting = self.registry.tools[name].side_effecting
                    if side_effecting:
                        problems.append(
                            f"call[{index}] '{name}' repeats call[{first}] with identical "
                            f"arguments, and the tool is side-effecting"
                        )
                    elif side_effecting is None:
                        notes.append(
                            f"call[{index}] '{name}' repeats call[{first}], but the registry "
                            f"does not declare whether '{name}' is side-effecting"
                        )
                else:
                    seen[fingerprint] = index

        prediction = [_summarize(call) for call in calls]
        passed = not problems
        explanation = "; ".join(problems) if problems else ("; ".join(notes) or None)

        return EvalResult(
            trace_id=trace.trace_id,
            evaluator_id=self.id,
            evaluator_version=self._version,
            score=1.0 if passed else 0.0,
            passed=passed,
            normalized_prediction=prediction,
            ground_truth=None,
            explanation=explanation,
        )

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
