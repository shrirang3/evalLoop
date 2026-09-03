"""The LLM question evaluator: render, ask, extract, compare.

The fourth step carries plan/001's argument. Without ground truth the judge's
answer is recorded and `passed` stays None - writing False there would turn an
uncalibrated opinion into a failure the feedback compiler could train on.
"""

from __future__ import annotations

from typing import Any

import pytest

from evalloop.contracts import (
    EvalContext,
    JudgeConfig,
    LLMQuestionSpec,
    RenderedPrompt,
    Trace,
)
from evalloop.evaluate import LLMQuestionEvaluator, render_prompt
from evalloop.judge import JudgeClient, MockProvider, Timeout

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["answer"],
}


def _trace(**overrides: Any) -> Trace:
    payload: dict[str, Any] = {
        "trace_id": "t1",
        "input": {"user_request": "I want a refund"},
        "output": {"text": "Sure, refunded."},
        "ground_truth": {},
    }
    payload.update(overrides)
    return Trace.model_validate(payload)


def _spec(**overrides: Any) -> LLMQuestionSpec:
    payload: dict[str, Any] = {
        "id": "policy_followed",
        "type": "llm_question",
        "question": "Customer: {{ request }}\nAgent: {{ reply }}\nDid the agent follow policy?",
        "inputs": {"request": "input.user_request", "reply": "output.text"},
        "response_schema": SCHEMA,
        "matcher": "boolean",
    }
    payload.update(overrides)
    return LLMQuestionSpec.model_validate(payload)


def _evaluator(
    provider: Any, spec: LLMQuestionSpec | None = None
) -> tuple[LLMQuestionEvaluator, EvalContext]:
    resolved = spec or _spec()
    client = JudgeClient(
        JudgeConfig(provider="mock", model="stub-1"),
        provider,
        system_prompt=resolved.system_prompt,
        questions=[resolved.question],
        response_schema=resolved.response_schema,
        sleep=lambda _: None,
    )
    return (
        LLMQuestionEvaluator(resolved, client.version_hash),
        EvalContext(run_id="r1", judge=client),
    )


# --- rendering ---


def test_inputs_are_pulled_by_path_into_the_template() -> None:
    prompt = render_prompt(_spec(), _trace())
    content = prompt.messages[0]["content"]
    assert "I want a refund" in content
    assert "Sure, refunded." in content


def test_a_missing_input_path_renders_as_none_rather_than_failing() -> None:
    """An optional field being absent is normal. The judge sees None and can
    say so; refusing to ask at all would lose the trace."""
    prompt = render_prompt(
        _spec(inputs={"request": "input.nonexistent", "reply": "output.text"}), _trace()
    )
    assert "None" in prompt.messages[0]["content"]


def test_system_prompt_is_rendered_too() -> None:
    spec = _spec(system_prompt="You audit transcripts in {{ request }}.")
    assert "I want a refund" in (render_prompt(spec, _trace()).system or "")


def test_a_template_variable_with_no_input_is_a_config_bug() -> None:
    """Rendering it as an empty string would silently ask the judge a different
    question than the one written down."""
    evaluator, ctx = _evaluator(MockProvider(), _spec(question="{{ undeclared }}?", inputs={}))
    result = evaluator.evaluate(_trace(), ctx)
    assert result.error is not None
    assert "undeclared" in result.error


# --- the calibrated path ---


def test_agreement_with_ground_truth_passes() -> None:
    evaluator, ctx = _evaluator(
        MockProvider(answers=[{"answer": True, "reason": "within policy"}]),
        _spec(ground_truth="ground_truth.policy_followed"),
    )
    result = evaluator.evaluate(_trace(ground_truth={"policy_followed": True}), ctx)

    assert result.passed is True
    assert result.normalized_prediction is True
    assert result.ground_truth is True
    assert result.explanation == "within policy"


def test_disagreement_with_ground_truth_fails() -> None:
    """This is the judgecard's raw material: one cell of the confusion matrix."""
    evaluator, ctx = _evaluator(
        MockProvider(answers=[{"answer": True}]),
        _spec(ground_truth="ground_truth.policy_followed"),
    )
    result = evaluator.evaluate(_trace(ground_truth={"policy_followed": False}), ctx)
    assert result.passed is False
    assert result.is_failure


# --- the uncalibrated path ---


def test_no_ground_truth_records_the_answer_without_a_verdict() -> None:
    evaluator, ctx = _evaluator(
        MockProvider(answers=[{"answer": True, "reason": "looks fine"}]),
        _spec(ground_truth="ground_truth.policy_followed"),
    )
    result = evaluator.evaluate(_trace(), ctx)

    assert result.normalized_prediction is True  # measured
    assert result.passed is None  # but not a verdict
    assert result.score is None
    assert not result.is_failure
    assert "not calibrated" in (result.explanation or "")


def test_no_ground_truth_path_configured_is_also_uncalibrated() -> None:
    """A holdout question has no label by design and must still be recorded."""
    evaluator, ctx = _evaluator(MockProvider(answers=[{"answer": False}]))
    result = evaluator.evaluate(_trace(), ctx)
    assert result.passed is None
    assert result.normalized_prediction is False
    assert "no ground truth configured" in (result.explanation or "")


def test_the_judges_reason_survives_alongside_the_calibration_note() -> None:
    evaluator, ctx = _evaluator(MockProvider(answers=[{"answer": True, "reason": "polite"}]))
    explanation = evaluator.evaluate(_trace(), ctx).explanation or ""
    assert "polite" in explanation
    assert "not calibrated" in explanation


# --- failure modes ---


def test_a_transport_failure_becomes_an_errored_result() -> None:
    evaluator, ctx = _evaluator(MockProvider(raise_error=Timeout("timed out")))
    result = evaluator.evaluate(_trace(), ctx)

    assert result.error is not None
    assert result.passed is None
    assert not result.is_failure  # a broken evaluation is not a bad model
    assert result.judge_config_hash is not None


def test_an_unusable_answer_is_invalid_output() -> None:
    evaluator, ctx = _evaluator(MockProvider(answers=["not json", "still not json"]))
    result = evaluator.evaluate(_trace(), ctx)

    assert result.invalid_output
    assert result.passed is None
    assert not result.is_failure


def test_a_well_formed_answer_missing_the_field_is_invalid_output() -> None:
    """Schema-valid JSON that does not carry an answer is a judge problem, not
    a model problem."""
    evaluator, ctx = _evaluator(MockProvider(answers=[{"verdict": True}, {"verdict": True}]))
    result = evaluator.evaluate(_trace(), ctx)
    assert result.invalid_output
    assert "answer" in (result.explanation or "")


def test_an_unreadable_boolean_is_invalid_output() -> None:
    evaluator, ctx = _evaluator(MockProvider(answers=[{"answer": "maybe"}, {"answer": "maybe"}]))
    assert evaluator.evaluate(_trace(), ctx).invalid_output


def test_no_judge_in_the_context_is_an_error() -> None:
    """Deterministic evaluators get judge=None, which is what keeps them
    ungameable. An llm_question with no judge is a wiring bug."""
    evaluator, _ = _evaluator(MockProvider())
    result = evaluator.evaluate(_trace(), EvalContext(run_id="r1"))
    assert result.error is not None
    assert "no judge" in result.error


# --- matchers ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(True, True), (False, False), ("true", True), ("no", False), ("PASS", True), (1, True)],
)
def test_boolean_matcher_accepts_the_shapes_models_actually_emit(raw: Any, expected: bool) -> None:
    evaluator, ctx = _evaluator(MockProvider(answers=[{"answer": raw}]))
    assert evaluator.evaluate(_trace(), ctx).normalized_prediction is expected


def test_exact_label_matcher() -> None:
    evaluator, ctx = _evaluator(
        MockProvider(answers=[{"answer": "empathetic"}]), _spec(matcher="exact_label")
    )
    assert evaluator.evaluate(_trace(), ctx).normalized_prediction == "empathetic"


def test_label_map_matcher_normalizes_synonyms() -> None:
    spec = _spec(
        matcher="label_map",
        matcher_options={"map": {"good": "pass", "great": "pass", "bad": "fail"}},
    )
    evaluator, ctx = _evaluator(MockProvider(answers=[{"answer": "great"}]), spec)
    assert evaluator.evaluate(_trace(), ctx).normalized_prediction == "pass"


def test_numeric_tolerance_matcher_reads_a_score() -> None:
    evaluator, ctx = _evaluator(
        MockProvider(answers=[{"answer": 4}]), _spec(matcher="numeric_tolerance")
    )
    assert evaluator.evaluate(_trace(), ctx).normalized_prediction == 4.0


def test_set_overlap_matcher_sorts_for_stability() -> None:
    evaluator, ctx = _evaluator(
        MockProvider(answers=[{"answer": ["b", "a"]}]), _spec(matcher="set_overlap")
    )
    assert evaluator.evaluate(_trace(), ctx).normalized_prediction == ["a", "b"]


def test_a_custom_answer_key() -> None:
    spec = _spec(matcher_options={"answer_key": "verdict"})
    evaluator, ctx = _evaluator(MockProvider(answers=[{"verdict": True}]), spec)
    assert evaluator.evaluate(_trace(), ctx).normalized_prediction is True


# --- provenance ---


def test_every_result_carries_the_judge_hash_and_cost() -> None:
    evaluator, ctx = _evaluator(MockProvider(answers=[{"answer": True}], cost_usd=0.003))
    result = evaluator.evaluate(_trace(), ctx)

    assert result.judge_config_hash is not None
    assert result.cost_usd == 0.003
    assert result.tokens_in == 100
    assert result.raw_output == {"answer": True}


def test_the_evaluator_version_covers_the_judge() -> None:
    """The same question asked of a different model is a different measurement,
    so swapping the model has to change this hash - otherwise results from
    before and after would compare as though nothing moved."""
    spec = _spec()
    first = LLMQuestionEvaluator(spec, "judge-hash-a").version_hash()
    second = LLMQuestionEvaluator(spec, "judge-hash-b").version_hash()
    assert first != second


def test_the_evaluator_version_covers_the_question() -> None:
    first = LLMQuestionEvaluator(_spec(), "j").version_hash()
    second = LLMQuestionEvaluator(_spec(question="Something else?"), "j").version_hash()
    assert first != second


def test_the_prompt_the_judge_receives_is_the_rendered_one() -> None:
    """The cache and PII redaction both key off the rendered prompt, so the
    template must not reach a provider."""
    provider = MockProvider(answers=[{"answer": True}])
    evaluator, ctx = _evaluator(provider)
    evaluator.evaluate(_trace(), ctx)

    sent: RenderedPrompt = provider.calls[0]
    assert "{{" not in sent.messages[0]["content"]


def test_an_unimplemented_matcher_is_named() -> None:
    """Rather than silently returning the raw value, which would compare
    against ground truth as a string and quietly pass or fail at random."""
    from evalloop.evaluate.llm.question import extract

    spec = _spec()
    object.__setattr__(spec, "matcher", "jaccard")  # bypass the frozen model
    with pytest.raises(ValueError, match="not implemented yet"):
        extract({"answer": "x"}, spec)


def test_the_mock_judge_can_depend_on_the_trace() -> None:
    """Needed for P3: a verbosity-biased judge answers differently for a long
    reply than a short one, which is the whole point of the bias probes."""

    def biased(prompt: RenderedPrompt) -> dict[str, Any]:
        long_enough = len(prompt.messages[0]["content"]) > 80
        return {"answer": long_enough, "reason": "length"}

    evaluator, ctx = _evaluator(MockProvider(answer_fn=biased))
    verbose = _trace(output={"text": "x" * 200})
    terse = _trace(output={"text": "no"})

    assert evaluator.evaluate(verbose, ctx).normalized_prediction is True
    assert evaluator.evaluate(terse, ctx).normalized_prediction is False
