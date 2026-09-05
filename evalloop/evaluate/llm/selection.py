"""Which tool *should* have been called - decided blind, compared afterwards.

`llm/question.py` shows the judge what the model did and asks whether it was
right. That is the anchoring failure `plan/001` section 6.2 forbids for human
labellers: a judge shown a decision rationalises it. This inverts the order.

    render     catalogue + request + policy. Never the call, never the reply.
    ask        answer constrained to this node's allowlist plus `none`
    compare    every called tool must be in the judge's `acceptable` set

The comparison is the same machinery `json_match` uses. What changes is where
the target came from: ground truth is a *stored* target, and this is a
*computed* one (`plan/002` section 8). So `EvalResult.ground_truth` holds the
judge's pick, with `judge_config_hash` set - which is what marks the difference
between a fact and an opinion when P4 reads these rows back.

Two properties fall out of the closed answer space and both matter more than
they look:

- a hallucinated answer is unrepresentable, not merely unlikely
- the task is classification over 5-15 labels, so a cheap model is enough, which
  is what makes running this over every production trace affordable
"""

from __future__ import annotations

from typing import Any

import jinja2

from evalloop.contracts.paths import Missing, resolve_path
from evalloop.contracts.protocols import EvalContext, RenderedPrompt
from evalloop.contracts.result import EvalResult
from evalloop.contracts.suite import ToolSelectionSpec
from evalloop.contracts.tools import NONE_CHOICE, ToolRegistry
from evalloop.contracts.trace import Trace, canonical_json
from evalloop.evaluate.base import error_result, not_applicable, version_of

__all__ = ["ToolSelectionEvaluator", "called_tools", "render_selection_prompt", "selection_schema"]

_ENVIRONMENT = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)

_TEMPLATE = """Available tools at node `{{ node }}`:

{{ catalogue }}
{% if policy %}
Policy:
{{ policy }}
{% endif %}
Customer said:
  {{ request }}
{% for name, value in extra %}
{{ name }}:
  {{ value }}
{% endfor %}
Which single tool should be called in response? If no tool should be called, \
answer "{{ none_choice }}".

`best` is your choice. `acceptable` lists every tool that would also be a \
defensible response, including `best`."""


def selection_schema(registry: ToolRegistry, node: str | None) -> dict[str, Any]:
    """A schema whose answer space is exactly this node's tools plus `none`."""
    choices = registry.choices(node)
    return {
        "type": "object",
        "properties": {
            "best": {"enum": choices},
            "acceptable": {"type": "array", "items": {"enum": choices}},
            "reason": {"type": "string"},
        },
        "required": ["best", "acceptable", "reason"],
    }


def render_selection_prompt(
    spec: ToolSelectionSpec,
    trace: Trace,
    registry: ToolRegistry,
    node: str | None,
) -> RenderedPrompt:
    """Build the prompt. The trace's own output never enters it."""
    extra: list[tuple[str, Any]] = []
    for name, path in sorted(spec.inputs.items()):
        value = resolve_path(trace, path)
        if not isinstance(value, Missing):
            extra.append((name, value if isinstance(value, str) else canonical_json(value)))

    request = resolve_path(trace, spec.request)
    question = _ENVIRONMENT.from_string(_TEMPLATE).render(
        node=node or "(unspecified)",
        catalogue=registry.catalogue(node),
        policy=spec.policy,
        request="" if isinstance(request, Missing) else request,
        extra=extra,
        none_choice=NONE_CHOICE,
    )
    return RenderedPrompt(
        system=spec.system_prompt,
        messages=[{"role": "user", "content": question}],
    )


def called_tools(calls: Any) -> list[str]:
    """Tool names actually called, deduplicated, in order. `[]` means none."""
    if not isinstance(calls, (list, tuple)):
        return []
    names: list[str] = []
    for call in calls:
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        if isinstance(name, str) and name not in names:
            names.append(name)
    return names


class ToolSelectionEvaluator:
    """One trace, one blind selection, one verdict."""

    def __init__(
        self,
        spec: ToolSelectionSpec,
        registry: ToolRegistry,
        judge_version_hash: str,
    ) -> None:
        if spec.samples != 1:
            raise ValueError(
                f"samples={spec.samples} is not implemented yet. Repeated sampling needs a "
                "cache key that varies per sample, which arrives with the P3a "
                "self-consistency probe; asking a temperature-0 judge the same question "
                "n times and calling the agreement 'consistency' would be measuring the cache"
            )

        self.spec = spec
        self.id = spec.id
        self.registry = registry
        self._judge_hash = judge_version_hash
        self._version = version_of(
            {
                **spec.version_payload(),
                "registry": registry.registry_hash(),
                "judge_version": judge_version_hash,
            }
        )

    def version_hash(self) -> str:
        """Covers the registry, because the catalogue *is* the prompt."""
        return self._version

    def evaluate(self, trace: Trace, ctx: EvalContext) -> EvalResult:
        if ctx.judge is None:
            return error_result(
                trace,
                self.id,
                self._version,
                RuntimeError("no judge available for a tool_selection evaluator"),
            )

        node = self._node(trace)
        if not self.registry.knows_node(node):
            # Not a model failure and not a pass: the suite is pointed at a node
            # the registry has never heard of, and guessing which tools are
            # allowed there would invent a verdict.
            return not_applicable(
                trace, self.id, self._version, f"node '{node}' is not in the registry"
            )

        try:
            prompt = render_selection_prompt(self.spec, trace, self.registry, node)
        except jinja2.UndefinedError as exc:
            return error_result(trace, self.id, self._version, exc)

        response = ctx.judge.ask(prompt, selection_schema(self.registry, node))

        if response.error is not None:
            return self._result(trace, response, error=response.error)
        if response.parsed is None:
            return self._result(trace, response, invalid=True, explanation=response.raw[:500])

        best = response.parsed.get("best")
        if not isinstance(best, str):
            return self._result(
                trace, response, invalid=True, explanation=f"no 'best' in answer {response.parsed}"
            )

        # Union `best` in: a judge that names a tool and omits it from its own
        # acceptable set has contradicted itself, and reading that as "the model
        # was wrong" would be a false failure.
        acceptable = {best}
        raw_acceptable = response.parsed.get("acceptable")
        if isinstance(raw_acceptable, list):
            acceptable.update(item for item in raw_acceptable if isinstance(item, str))

        called = called_tools(resolve_path(trace, self.spec.actual))
        # Calling nothing is an answer, and `none` is how the judge says it.
        observed = called or [NONE_CHOICE]

        offenders = [name for name in observed if name not in acceptable]
        passed = not offenders

        return self._result(
            trace,
            response,
            prediction=observed,
            truth=best,
            passed=passed,
            explanation=_explain(
                offenders, best, sorted(acceptable), response.parsed.get("reason")
            ),
        )

    def _node(self, trace: Trace) -> str | None:
        """Node from the spec's path, else from the first call that names one.

        Reading it off a call is safe here even though the calls are withheld
        from the judge: which node ran is routing, not the decision under test.
        """
        if self.spec.node_path is not None:
            value = resolve_path(trace, self.spec.node_path)
            if isinstance(value, str):
                return value
        for call in trace.output.tool_calls:
            if call.node is not None:
                return call.node
        return None

    def _result(
        self,
        trace: Trace,
        response: Any,
        *,
        prediction: Any = None,
        truth: Any = None,
        passed: bool | None = None,
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
            ground_truth=truth,
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


def _explain(
    offenders: list[str],
    best: str,
    acceptable: list[str],
    reason: str | None,
) -> str | None:
    if not offenders:
        return reason
    called = ", ".join(offenders)
    return f"called {called}; judge chose {best} (acceptable: {', '.join(acceptable)})" + (
        f" - {reason}" if reason else ""
    )
