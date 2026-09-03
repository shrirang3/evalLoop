"""Ask a judge one question about a trace.

Four steps, and the fourth is the one that carries plan/001's argument:

    render     Jinja2 over the question, with `inputs` pulling values by path
    ask        schema-constrained, via the judge client
    extract    a matcher turns the answer into a normalized prediction
    compare    against ground truth *if there is any*

Without ground truth the answer is still recorded, and `passed` stays None. The
judge's opinion is a measurement either way; what it is not, uncalibrated, is a
verdict. Writing False there would turn an unvalidated opinion into a failure
the feedback compiler could train on.
"""

from __future__ import annotations

from typing import Any

import jinja2

from evalloop.contracts.paths import MISSING, Missing, resolve_path
from evalloop.contracts.protocols import EvalContext, RenderedPrompt
from evalloop.contracts.result import EvalResult
from evalloop.contracts.suite import LLMQuestionSpec
from evalloop.contracts.trace import Trace
from evalloop.evaluate.base import error_result, version_of

__all__ = ["LLMQuestionEvaluator", "extract", "render_prompt"]

# StrictUndefined: an `inputs` key the template never uses is harmless, but a
# template variable with no `inputs` entry is a config bug. Rendering it as an
# empty string would silently ask the judge a different question than intended.
_ENVIRONMENT = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)

_DEFAULT_ANSWER_KEY = "answer"


def render_prompt(spec: LLMQuestionSpec, trace: Trace) -> RenderedPrompt:
    """Fill the question template from the trace."""
    variables: dict[str, Any] = {}
    for name, path in spec.inputs.items():
        value = resolve_path(trace, path)
        variables[name] = None if isinstance(value, Missing) else value

    question = _ENVIRONMENT.from_string(spec.question).render(**variables)
    system = (
        _ENVIRONMENT.from_string(spec.system_prompt).render(**variables)
        if spec.system_prompt
        else None
    )
    return RenderedPrompt(system=system, messages=[{"role": "user", "content": question}])


def extract(parsed: dict[str, Any], spec: LLMQuestionSpec) -> Any:
    """Turn the judge's structured answer into a comparable value."""
    key = str(spec.matcher_options.get("answer_key", _DEFAULT_ANSWER_KEY))
    if key not in parsed:
        raise KeyError(f"judge answer has no {key!r} field; got keys {sorted(parsed)}")
    raw = parsed[key]

    if spec.matcher == "boolean":
        return _as_bool(raw)
    if spec.matcher == "exact_label":
        return str(raw)
    if spec.matcher == "label_map":
        mapping: dict[str, Any] = spec.matcher_options.get("map", {})
        return mapping.get(str(raw), str(raw))
    if spec.matcher == "numeric_tolerance":
        return float(raw)
    if spec.matcher == "set_overlap":
        return sorted(str(item) for item in raw)
    raise ValueError(f"matcher {spec.matcher!r} is not implemented yet")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "pass"}:
        return True
    if text in {"false", "no", "n", "0", "fail"}:
        return False
    raise ValueError(f"cannot read {value!r} as a boolean answer")


class LLMQuestionEvaluator:
    """One question, one judge, one result per trace."""

    def __init__(self, spec: LLMQuestionSpec, judge_version_hash: str) -> None:
        self.spec = spec
        self.id = spec.id
        self._version = version_of({**spec.version_payload(), "judge_version": judge_version_hash})
        self._judge_hash = judge_version_hash

    def version_hash(self) -> str:
        """Covers the judge as well as the question.

        The same question asked of a different model is a different measurement,
        so a model swap has to change this hash - otherwise results from before
        and after would compare as though nothing moved.
        """
        return self._version

    def evaluate(self, trace: Trace, ctx: EvalContext) -> EvalResult:
        if ctx.judge is None:
            return error_result(
                trace,
                self.id,
                self._version,
                RuntimeError("no judge available for an llm_question evaluator"),
            )

        try:
            prompt = render_prompt(self.spec, trace)
        except jinja2.UndefinedError as exc:
            return error_result(trace, self.id, self._version, exc)

        response = ctx.judge.ask(prompt, self.spec.response_schema)

        if response.error is not None:
            return EvalResult(
                trace_id=trace.trace_id,
                evaluator_id=self.id,
                evaluator_version=self._version,
                judge_config_hash=self._judge_hash,
                error=response.error,
                latency_ms=response.latency_ms,
                cost_usd=response.usage.cost_usd,
                tokens_in=response.usage.tokens_in,
                tokens_out=response.usage.tokens_out,
                cache_hit=response.cache_hit,
            )

        if response.parsed is None:
            # The judge answered, unusably. Recorded rather than retried again:
            # this rate is a headline number on the judgecard.
            return self._result(
                trace, response, prediction=None, invalid=True, explanation=response.raw[:500]
            )

        try:
            prediction = extract(response.parsed, self.spec)
        except (KeyError, ValueError, TypeError) as exc:
            return self._result(
                trace, response, prediction=None, invalid=True, explanation=str(exc)
            )

        truth = resolve_path(trace, self.spec.ground_truth) if self.spec.ground_truth else MISSING
        explanation = response.parsed.get("reason")

        if isinstance(truth, Missing):
            # Measured, reported, and explicitly not a verdict.
            result = self._result(trace, response, prediction=prediction, explanation=explanation)
            return result.model_copy(
                update={
                    "explanation": _uncalibrated(explanation, self.spec.ground_truth),
                }
            )

        passed = prediction == truth
        return self._result(
            trace,
            response,
            prediction=prediction,
            truth=truth,
            passed=passed,
            explanation=explanation,
        )

    def _result(
        self,
        trace: Trace,
        response: Any,
        *,
        prediction: Any,
        truth: Any = None,
        passed: bool | None = None,
        invalid: bool = False,
        explanation: str | None = None,
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
            latency_ms=response.latency_ms,
            cost_usd=response.usage.cost_usd,
            tokens_in=response.usage.tokens_in,
            tokens_out=response.usage.tokens_out,
        )


def _uncalibrated(reason: str | None, path: str | None) -> str:
    note = (
        "measured but not calibrated: no ground truth configured"
        if path is None
        else f"measured but not calibrated: no ground truth at {path}"
    )
    return f"{reason} [{note}]" if reason else note
