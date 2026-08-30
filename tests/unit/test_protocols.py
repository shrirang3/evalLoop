"""The Evaluator and Judge protocols are structural: anything with the right
shape qualifies, without importing a base class from EvalLoop."""

from __future__ import annotations

from typing import Any

from evalloop.contracts import (
    EvalContext,
    EvalResult,
    Evaluator,
    Judge,
    JudgeResponse,
    RenderedPrompt,
    Trace,
)


class AlwaysPasses:
    """A customer's own check, written without knowing EvalLoop's internals."""

    id = "always_passes"

    def version_hash(self) -> str:
        return "static-v1"

    def evaluate(self, trace: Trace, ctx: EvalContext) -> EvalResult:
        return EvalResult(
            trace_id=trace.trace_id,
            evaluator_id=self.id,
            evaluator_version=self.version_hash(),
            passed=True,
        )


class StubJudge:
    def ask(self, prompt: RenderedPrompt, schema: dict[str, Any]) -> JudgeResponse:
        return JudgeResponse(raw='{"answer": true}', parsed={"answer": True}, judge_config_hash="h")


def test_duck_typed_classes_satisfy_the_protocols() -> None:
    assert isinstance(AlwaysPasses(), Evaluator)
    assert isinstance(StubJudge(), Judge)


def test_deterministic_evaluators_get_no_judge() -> None:
    """A deterministic check with no judge is what makes its result ungameable
    by a training loop optimising against that judge (plan/001 section 3.2)."""
    ctx = EvalContext(run_id="r1")
    assert ctx.judge is None

    trace = Trace.model_validate({"trace_id": "t1", "input": {}, "output": {"text": "hi"}})
    result = AlwaysPasses().evaluate(trace, ctx)
    assert result.passed is True
    assert result.judge_config_hash is None


def test_rendered_prompt_cache_payload_is_the_post_render_form() -> None:
    """The cache and PII redaction both key off the rendered prompt, not the
    template, so redaction cannot be bypassed by a template change."""
    p = RenderedPrompt(system="be terse", messages=[{"role": "user", "content": "hi"}])
    assert p.cache_key_payload() == {
        "system": "be terse",
        "messages": [{"role": "user", "content": "hi"}],
    }
