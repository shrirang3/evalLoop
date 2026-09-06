"""Does the reply describe what the calls actually did?

The third check in `plan/002` section 2, and the only one that reads the model's
prose. It exists because tool correctness and text correctness are separate
axes, and the other two checks see only the first:

    tools right, text right   pass
    tools right, text wrong   registry check passes, selection passes, and the
                              customer has still been told they got a refund
    tools wrong, text either  the other two catch it

Two properties make this cheap. It needs no ground truth, and it needs no
computed target either - unlike `tool_selection`, the correct answer is known in
advance. A reply should always match its calls, so consistency *is* the target
and a disagreement is a verdict rather than an opinion awaiting calibration.

The judge is deliberately steered away from the false-failure direction
(`plan/001` section 5.3): a reply may be warmer, longer, or more apologetic than
the calls without contradicting them. Only claims about *actions* count.
"""

from __future__ import annotations

from typing import Any

from evalloop.contracts.paths import Missing, resolve_path
from evalloop.contracts.protocols import EvalContext, RenderedPrompt
from evalloop.contracts.result import EvalResult
from evalloop.contracts.suite import TextMatchesToolsSpec
from evalloop.contracts.tools import ToolRegistry
from evalloop.contracts.trace import Trace, canonical_json
from evalloop.evaluate.base import error_result, not_applicable, version_of

__all__ = [
    "CONSISTENCY_SCHEMA",
    "TextMatchesToolsEvaluator",
    "format_calls",
    "render_consistency_prompt",
]

CONSISTENCY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "consistent": {"type": "boolean"},
        "contradiction": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["consistent", "reason"],
}

NO_CALLS = "  (the agent called no tools)"

_DEFAULT_SYSTEM = (
    "You check whether a support agent's reply matches the tool calls it made. "
    "Judge only claims about actions: what was done, what will happen to the "
    "customer's order or money. Warmth, apology, length and extra detail are not "
    "contradictions. A reply that describes an action no call performed, or "
    "describes a different action than the calls performed, is inconsistent."
)

_TEMPLATE = """The agent called:
{calls}

The agent replied:
  {text}

Does the reply accurately describe what those calls did? Answer `consistent: false` \
only when the reply claims an action the calls did not perform, or describes a \
different action than they performed. Put the offending phrase in `contradiction`."""


def format_calls(calls: Any, registry: ToolRegistry | None, describe: bool) -> str:
    """Render calls for a prompt: `issue_refund(order_id="ORD-42", amount=79.99)`.

    Arguments go through `canonical_json` rather than `str`, which is what keeps
    a Python repr - `ToolCall(name=..., call_id=None)` - out of the judge's view.
    """
    if not isinstance(calls, (list, tuple)) or not calls:
        return NO_CALLS

    lines: list[str] = []
    for call in calls:
        name = _attr(call, "name")
        if not isinstance(name, str):
            continue
        arguments = _attr(call, "arguments") or {}
        rendered = (
            ", ".join(f"{key}={canonical_json(value)}" for key, value in sorted(arguments.items()))
            if isinstance(arguments, dict)
            else ""
        )
        lines.append(f"  {name}({rendered})")
        if describe and registry is not None and registry.has(name):
            lines.append(f"      {registry.tools[name].description}")

    return "\n".join(lines) or NO_CALLS


def render_consistency_prompt(
    spec: TextMatchesToolsSpec,
    trace: Trace,
    registry: ToolRegistry | None,
) -> RenderedPrompt:
    text = resolve_path(trace, spec.text)
    calls = resolve_path(trace, spec.actual)
    question = _TEMPLATE.format(
        calls=format_calls(
            None if isinstance(calls, Missing) else calls, registry, spec.describe_tools
        ),
        text="" if isinstance(text, Missing) or text is None else text,
    )
    return RenderedPrompt(
        system=spec.system_prompt or _DEFAULT_SYSTEM,
        messages=[{"role": "user", "content": question}],
    )


class TextMatchesToolsEvaluator:
    """One trace, one consistency verdict."""

    def __init__(
        self,
        spec: TextMatchesToolsSpec,
        judge_version_hash: str,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.spec = spec
        self.id = spec.id
        self.registry = registry
        self._judge_hash = judge_version_hash
        self._version = version_of(
            {
                **spec.version_payload(),
                "judge_version": judge_version_hash,
                # Descriptions are prompt text when they are included, and are
                # not part of the measurement when they are not.
                "registry": (
                    registry.registry_hash()
                    if registry is not None and spec.describe_tools
                    else None
                ),
            }
        )

    def version_hash(self) -> str:
        return self._version

    def evaluate(self, trace: Trace, ctx: EvalContext) -> EvalResult:
        if ctx.judge is None:
            return error_result(
                trace,
                self.id,
                self._version,
                RuntimeError("no judge available for a text_matches_tools evaluator"),
            )

        text = resolve_path(trace, self.spec.text)
        if isinstance(text, Missing) or not (isinstance(text, str) and text.strip()):
            # Nothing was said, so nothing can contradict the calls. Not a pass:
            # the check never looked at this trace.
            return not_applicable(
                trace, self.id, self._version, f"no reply text at {self.spec.text}"
            )

        calls = resolve_path(trace, self.spec.actual)
        # An empty call list is *not* skipped. "I've opened a warranty claim"
        # with no calls at all is the exact failure this check exists to catch,
        # and abstaining there would hide it.

        prompt = render_consistency_prompt(self.spec, trace, self.registry)
        response = ctx.judge.ask(prompt, CONSISTENCY_SCHEMA)

        if response.error is not None:
            return self._result(trace, response, error=response.error)
        if response.parsed is None:
            return self._result(trace, response, invalid=True, explanation=response.raw[:500])

        consistent = response.parsed.get("consistent")
        if not isinstance(consistent, bool):
            return self._result(
                trace,
                response,
                invalid=True,
                explanation=f"no boolean 'consistent' in answer {response.parsed}",
            )

        return self._result(
            trace,
            response,
            passed=consistent,
            prediction=_called_names(calls),
            explanation=_explain(response.parsed)
            if not consistent
            else response.parsed.get("reason"),
        )

    def _result(
        self,
        trace: Trace,
        response: Any,
        *,
        passed: bool | None = None,
        prediction: Any = None,
        invalid: bool = False,
        explanation: str | None = None,
        error: str | None = None,
    ) -> EvalResult:
        return EvalResult(
            trace_id=trace.trace_id,
            evaluator_id=self.id,
            evaluator_version=self._version,
            score=None if passed is None else (1.0 if passed else 0.0),
            passed=passed,
            normalized_prediction=prediction,
            # No target, stored or computed: "the reply matches the calls" is
            # the answer by construction, so there is nothing to record here.
            ground_truth=None,
            explanation=explanation,
            raw_output=response.parsed,
            judge_config_hash=self._judge_hash,
            cache_hit=response.cache_hit,
            invalid_output=invalid,
            error=error,
            latency_ms=response.latency_ms,
            cost_usd=response.usage.cost_usd,
            tokens_in=response.usage.tokens_in,
            tokens_out=response.usage.tokens_out,
        )


def _explain(parsed: dict[str, Any]) -> str | None:
    reason = parsed.get("reason")
    contradiction = parsed.get("contradiction")
    if isinstance(contradiction, str) and contradiction.strip():
        return f"reply says {contradiction!r}" + (f" - {reason}" if reason else "")
    return reason if isinstance(reason, str) else None


def _called_names(calls: Any) -> list[str]:
    if not isinstance(calls, (list, tuple)):
        return []
    names = [_attr(call, "name") for call in calls]
    return [name for name in names if isinstance(name, str)]


def _attr(call: Any, name: str) -> Any:
    if isinstance(call, dict):
        return call.get(name)
    return getattr(call, name, None)
