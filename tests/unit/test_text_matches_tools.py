"""Does the reply describe what the calls actually did?

The axis the other two tool checks cannot see. `tool_registry_check` says the
call was legal, `tool_selection` says it was the right call, and the customer
has still been told they got a refund they did not get.
"""

from __future__ import annotations

from typing import Any

import pytest

from evalloop.contracts import EvalContext, JudgeConfig, ToolRegistry, Trace
from evalloop.contracts.suite import TextMatchesToolsSpec
from evalloop.evaluate import TextMatchesToolsEvaluator, render_consistency_prompt
from evalloop.evaluate.llm.consistency import CONSISTENCY_SCHEMA
from evalloop.judge import JudgeClient, MockProvider

REGISTRY = ToolRegistry.model_validate(
    {
        "tools": {
            "issue_refund": {
                "description": "Refund an order to the original payment method. Irreversible.",
                "arguments": {"order_id": {"type": "string", "required": True}},
                "side_effecting": True,
            },
            "open_warranty_claim": {
                "description": "Open a replacement claim for a damaged or faulty item.",
                "arguments": {"order_id": {"type": "string", "required": True}},
                "side_effecting": True,
            },
        }
    }
)

WARRANTY_CALL = {"name": "open_warranty_claim", "arguments": {"order_id": "ORD-8891"}}


def _trace(text: str | None = "Sure, I've processed your full refund.", **overrides: Any) -> Trace:
    payload: dict[str, Any] = {
        "trace_id": "sb-0417",
        "input": {"user_request": "blender arrived broken"},
        "output": {"text": text, "tool_calls": [WARRANTY_CALL]},
    }
    output = payload["output"] | overrides.pop("output", {})
    payload["output"] = output
    payload.update(overrides)
    return Trace.model_validate(payload)


def _evaluator(
    answer: Any,
    spec: TextMatchesToolsSpec | None = None,
    registry: ToolRegistry | None = REGISTRY,
) -> tuple[TextMatchesToolsEvaluator, EvalContext, MockProvider]:
    resolved = spec or TextMatchesToolsSpec(id="text_matches_tools")
    provider = MockProvider(answers=[answer])
    client = JudgeClient(
        JudgeConfig(provider="mock", model="stub-1"),
        provider,
        system_prompt=resolved.system_prompt,
        questions=[resolved.type],
        response_schema=CONSISTENCY_SCHEMA,
        sleep=lambda _: None,
    )
    return (
        TextMatchesToolsEvaluator(resolved, client.version_hash, registry),
        EvalContext(run_id="r1", judge=client),
        provider,
    )


# --- the prompt ---


def test_the_prompt_shows_both_the_calls_and_the_reply() -> None:
    """Unlike tool_selection, which withholds the call. This check is the pair."""
    content = render_consistency_prompt(TextMatchesToolsSpec(id="c"), _trace(), REGISTRY).messages[
        0
    ]["content"]
    assert "open_warranty_claim" in content
    assert "processed your full refund" in content


def test_arguments_render_as_json_not_a_python_repr() -> None:
    content = render_consistency_prompt(TextMatchesToolsSpec(id="c"), _trace(), REGISTRY).messages[
        0
    ]["content"]
    assert 'order_id="ORD-8891"' in content
    assert "ToolCall(" not in content
    assert "call_id" not in content


def test_registry_descriptions_are_included_when_available() -> None:
    """A judge told that a refund is irreversible reads a hedged reply differently."""
    content = render_consistency_prompt(TextMatchesToolsSpec(id="c"), _trace(), REGISTRY).messages[
        0
    ]["content"]
    assert "Open a replacement claim" in content


def test_descriptions_can_be_turned_off() -> None:
    content = render_consistency_prompt(
        TextMatchesToolsSpec(id="c", describe_tools=False), _trace(), REGISTRY
    ).messages[0]["content"]
    assert "Open a replacement claim" not in content
    assert "open_warranty_claim" in content


def test_it_works_with_no_registry_at_all() -> None:
    """Names alone are enough to ask the question; a registry only sharpens it."""
    content = render_consistency_prompt(TextMatchesToolsSpec(id="c"), _trace(), None).messages[0][
        "content"
    ]
    assert "open_warranty_claim" in content


def test_no_calls_is_stated_rather_than_left_blank() -> None:
    content = render_consistency_prompt(
        TextMatchesToolsSpec(id="c"), _trace(output={"tool_calls": []}), REGISTRY
    ).messages[0]["content"]
    assert "called no tools" in content


# --- the verdict ---


def test_a_reply_contradicting_the_calls_fails() -> None:
    """sb-0417: warranty claim opened, customer told they were refunded."""
    evaluator, ctx, _ = _evaluator(
        {
            "consistent": False,
            "contradiction": "I've processed your full refund",
            "reason": "the call opened a warranty claim",
        }
    )
    result = evaluator.evaluate(_trace(), ctx)
    assert result.passed is False
    assert "processed your full refund" in (result.explanation or "")


def test_a_matching_reply_passes() -> None:
    evaluator, ctx, _ = _evaluator({"consistent": True, "reason": "matches"})
    assert evaluator.evaluate(_trace("I've opened a warranty claim."), ctx).passed is True


def test_there_is_no_target_because_consistency_is_the_target() -> None:
    """No stored ground truth and no computed one - unlike tool_selection, the
    correct answer is known in advance."""
    evaluator, ctx, _ = _evaluator({"consistent": True, "reason": "matches"})
    result = evaluator.evaluate(_trace(), ctx)
    assert result.ground_truth is None
    assert result.judge_config_hash is not None


def test_a_claim_with_no_calls_at_all_is_judged_not_skipped() -> None:
    """ "I've opened a warranty claim" with an empty call list is exactly the
    failure this check exists to catch."""
    evaluator, ctx, provider = _evaluator(
        {"consistent": False, "contradiction": "I've opened a claim", "reason": "no call was made"}
    )
    result = evaluator.evaluate(
        _trace("I've opened a warranty claim.", output={"tool_calls": []}), ctx
    )
    assert result.passed is False
    assert provider.calls  # the judge was actually asked


def test_a_trace_with_no_reply_is_not_applicable() -> None:
    """Nothing was said, so nothing can contradict the calls - and no judge call
    is paid for."""
    evaluator, ctx, provider = _evaluator({"consistent": True, "reason": "x"})
    result = evaluator.evaluate(_trace(None), ctx)
    assert result.passed is None
    assert "no reply text" in (result.explanation or "")
    assert provider.calls == []


def test_a_blank_reply_is_also_not_applicable() -> None:
    evaluator, ctx, _ = _evaluator({"consistent": True, "reason": "x"})
    assert evaluator.evaluate(_trace("   "), ctx).passed is None


# --- degrading honestly ---


def test_a_non_boolean_answer_is_invalid_not_a_failure() -> None:
    evaluator, ctx, _ = _evaluator({"consistent": "maybe", "reason": "x"})
    result = evaluator.evaluate(_trace(), ctx)
    assert result.invalid_output is True
    assert result.passed is None
    assert result.is_failure is False


def test_an_unparseable_answer_is_invalid() -> None:
    evaluator, ctx, _ = _evaluator("not json")
    assert evaluator.evaluate(_trace(), ctx).invalid_output is True


def test_no_judge_is_an_error_not_a_verdict() -> None:
    evaluator, _, _ = _evaluator({"consistent": True, "reason": "x"})
    result = evaluator.evaluate(_trace(), EvalContext(run_id="r1"))
    assert result.error is not None
    assert result.passed is None


def test_version_hash_covers_descriptions_only_when_they_are_shown() -> None:
    """Registry text is part of the measurement when it reaches the prompt, and
    not part of it when it does not."""
    edited = REGISTRY.model_dump()
    edited["tools"]["open_warranty_claim"]["description"] = "Something else entirely."
    other = ToolRegistry.model_validate(edited)

    shown = TextMatchesToolsSpec(id="c")
    hidden = TextMatchesToolsSpec(id="c", describe_tools=False)

    assert (
        TextMatchesToolsEvaluator(shown, "j", REGISTRY).version_hash()
        != TextMatchesToolsEvaluator(shown, "j", other).version_hash()
    )
    assert (
        TextMatchesToolsEvaluator(hidden, "j", REGISTRY).version_hash()
        == TextMatchesToolsEvaluator(hidden, "j", other).version_hash()
    )


@pytest.mark.parametrize("field", ["text", "actual"])
def test_paths_are_configurable(field: str) -> None:
    spec = TextMatchesToolsSpec.model_validate({"id": "c", field: "metadata.custom"})
    assert getattr(spec, field) == "metadata.custom"
