"""Select-mode tool judging.

The property under test throughout: the judge decides *before* seeing the
decision. Everything else here - the closed answer space, `none` as an answer,
`acceptable` as a set - exists to keep a blind judgement from producing false
failures, which is the expensive direction (plan/001 section 5.3).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from evalloop.contracts import EvalContext, JudgeConfig, ToolRegistry, Trace
from evalloop.contracts.suite import ToolSelectionSpec
from evalloop.evaluate import ToolSelectionEvaluator, render_selection_prompt, selection_schema
from evalloop.judge import JudgeClient, MockProvider

REGISTRY = ToolRegistry.model_validate(
    {
        "tools": {
            "issue_refund": {
                "description": "Refund an order to the original payment method. Irreversible.",
                "arguments": {
                    "order_id": {"type": "string", "required": True},
                    "amount": {"type": "number", "required": True},
                },
                "side_effecting": True,
            },
            "open_warranty_claim": {
                "description": "Open a replacement claim for a damaged or faulty item.",
                "arguments": {"order_id": {"type": "string", "required": True}},
                "side_effecting": True,
            },
            "lookup_order": {
                "description": "Read-only order details.",
                "arguments": {"order_id": {"type": "string", "required": True}},
                "side_effecting": False,
            },
        },
        "nodes": {
            "refunds": {"tools": ["issue_refund", "open_warranty_claim", "lookup_order"]},
            "triage": {"tools": ["lookup_order"]},
        },
    }
)

REFUND_CALL = {
    "name": "issue_refund",
    "arguments": {"order_id": "ORD-8891", "amount": 79.99},
    "node": "refunds",
}


def _trace(calls: list[dict[str, Any]] | None = None, **overrides: Any) -> Trace:
    payload: dict[str, Any] = {
        "trace_id": "sb-0417",
        "input": {"user_request": "I ordered a blender 45 days ago and it arrived broken."},
        "output": {
            "text": "Sure, I've processed your full refund.",
            "tool_calls": [REFUND_CALL] if calls is None else calls,
        },
    }
    payload.update(overrides)
    return Trace.model_validate(payload)


def _spec(**overrides: Any) -> ToolSelectionSpec:
    payload: dict[str, Any] = {
        "id": "tool_selection",
        "type": "tool_selection",
        "policy": "Refunds are permitted within 30 days of purchase.",
    }
    payload.update(overrides)
    return ToolSelectionSpec.model_validate(payload)


def _evaluator(
    answer: Any,
    spec: ToolSelectionSpec | None = None,
) -> tuple[ToolSelectionEvaluator, EvalContext, MockProvider]:
    resolved = spec or _spec()
    provider = MockProvider(answers=[answer])
    client = JudgeClient(
        JudgeConfig(provider="mock", model="stub-1"),
        provider,
        system_prompt=resolved.system_prompt,
        questions=[REGISTRY.catalogue(), resolved.policy or ""],
        response_schema=selection_schema(REGISTRY, None),
        sleep=lambda _: None,
    )
    return (
        ToolSelectionEvaluator(resolved, REGISTRY, client.version_hash),
        EvalContext(run_id="r1", judge=client),
        provider,
    )


# --- the prompt is blind ---


def test_the_prompt_never_contains_the_call_or_the_reply() -> None:
    """The whole point. A judge shown the decision rationalises it."""
    content = render_selection_prompt(_spec(), _trace(), REGISTRY, "refunds").messages[0]["content"]
    assert "issue_refund" in content  # as a catalogue entry
    assert "ORD-8891" not in content  # not as the call that happened
    assert "processed your full refund" not in content


def test_inputs_that_read_the_model_output_are_refused() -> None:
    with pytest.raises(ValidationError, match="anchoring"):
        _spec(inputs={"reply": "output.text"})


def test_the_catalogue_is_scoped_to_the_node() -> None:
    content = render_selection_prompt(_spec(), _trace(), REGISTRY, "triage").messages[0]["content"]
    assert "lookup_order" in content
    assert "issue_refund" not in content


def test_the_answer_space_is_the_allowlist_plus_none() -> None:
    schema = selection_schema(REGISTRY, "triage")
    assert schema["properties"]["best"]["enum"] == ["lookup_order", "none"]


def test_extra_inputs_render_as_json_not_python_reprs() -> None:
    content = render_selection_prompt(
        _spec(inputs={"tier": "metadata.tier"}),
        _trace(metadata={"tier": {"plan": "premium"}}),
        REGISTRY,
        "refunds",
    ).messages[0]["content"]
    assert '{"plan":"premium"}' in content


# --- the verdict ---


def test_a_call_outside_the_acceptable_set_fails() -> None:
    evaluator, ctx, _ = _evaluator(
        {
            "best": "open_warranty_claim",
            "acceptable": ["open_warranty_claim", "lookup_order"],
            "reason": "outside the refund window, but damaged",
        }
    )
    result = evaluator.evaluate(_trace(), ctx)
    assert result.passed is False
    assert result.ground_truth == "open_warranty_claim"
    assert result.normalized_prediction == ["issue_refund"]
    assert "called issue_refund" in (result.explanation or "")


def test_the_judges_pick_is_stored_as_the_target() -> None:
    """A computed target, not a stored one - `judge_config_hash` is what marks
    the difference between a fact and an opinion when P4 reads the row back."""
    evaluator, ctx, _ = _evaluator(
        {"best": "open_warranty_claim", "acceptable": ["open_warranty_claim"], "reason": "r"}
    )
    result = evaluator.evaluate(_trace(), ctx)
    assert result.ground_truth == "open_warranty_claim"
    assert result.judge_config_hash is not None


def test_a_defensible_alternative_passes() -> None:
    """Set membership, not equality: several tools are often reasonable, and
    failing the model for picking the second-best one is a false failure."""
    evaluator, ctx, _ = _evaluator(
        {"best": "lookup_order", "acceptable": ["lookup_order", "issue_refund"], "reason": "either"}
    )
    assert evaluator.evaluate(_trace(), ctx).passed is True


def test_best_is_always_acceptable_even_if_the_judge_omits_it() -> None:
    """A judge that names a tool and leaves it out of its own acceptable set has
    contradicted itself; reading that as a model failure would be wrong."""
    evaluator, ctx, _ = _evaluator(
        {"best": "issue_refund", "acceptable": ["lookup_order"], "reason": "r"}
    )
    assert evaluator.evaluate(_trace(), ctx).passed is True


def test_calling_nothing_when_nothing_was_needed_passes() -> None:
    evaluator, ctx, _ = _evaluator({"best": "none", "acceptable": ["none"], "reason": "refuse"})
    assert evaluator.evaluate(_trace(calls=[]), ctx).passed is True


def test_calling_nothing_when_a_tool_was_needed_fails() -> None:
    """The missing-call case: `output.tool_calls` is empty and the judge picked
    a tool. Without `none` in the answer space this trace is invisible."""
    evaluator, ctx, _ = _evaluator(
        {"best": "lookup_order", "acceptable": ["lookup_order"], "reason": "look it up"}
    )
    result = evaluator.evaluate(_trace(calls=[]), ctx)
    assert result.passed is False
    assert result.normalized_prediction == ["none"]


def test_an_extra_call_alongside_a_correct_one_fails() -> None:
    """sb-0421: cancelled correctly, then also refunded. Every called tool has
    to be acceptable, not just one of them."""
    evaluator, ctx, _ = _evaluator(
        {"best": "lookup_order", "acceptable": ["lookup_order"], "reason": "just look"}
    )
    result = evaluator.evaluate(
        _trace(calls=[{"name": "lookup_order", "arguments": {"order_id": "O1"}}, REFUND_CALL]), ctx
    )
    assert result.passed is False
    assert result.normalized_prediction == ["lookup_order", "issue_refund"]


def test_repeated_calls_of_the_same_tool_count_once() -> None:
    """Duplicate detection is tool_registry_check's job, not this one's."""
    evaluator, ctx, _ = _evaluator(
        {"best": "issue_refund", "acceptable": ["issue_refund"], "reason": "r"}
    )
    result = evaluator.evaluate(_trace(calls=[REFUND_CALL, dict(REFUND_CALL)]), ctx)
    assert result.passed is True
    assert result.normalized_prediction == ["issue_refund"]


# --- degrading honestly ---


def test_an_unknown_node_is_not_applicable_rather_than_a_guess() -> None:
    evaluator, ctx, provider = _evaluator(
        {"best": "issue_refund", "acceptable": ["issue_refund"], "reason": "r"}
    )
    result = evaluator.evaluate(
        _trace(calls=[{**REFUND_CALL, "node": "billing"}]),
        ctx,
    )
    assert result.passed is None
    assert "not in the registry" in (result.explanation or "")
    assert provider.calls == []  # and no judge call was paid for


def test_an_unparseable_answer_is_invalid_not_a_failure() -> None:
    evaluator, ctx, _ = _evaluator("not json at all")
    result = evaluator.evaluate(_trace(), ctx)
    assert result.invalid_output is True
    assert result.passed is None
    assert result.is_failure is False


def test_no_judge_is_an_error_not_a_verdict() -> None:
    evaluator, _, _ = _evaluator({"best": "none", "acceptable": ["none"], "reason": "r"})
    result = evaluator.evaluate(_trace(), EvalContext(run_id="r1"))
    assert result.error is not None
    assert result.passed is None


def test_repeated_sampling_is_refused_until_it_can_vary_the_cache_key() -> None:
    """Asking a temperature-0 judge the same question n times through a cache
    measures the cache, not the judge."""
    with pytest.raises(ValueError, match="not implemented"):
        _evaluator(
            {"best": "none", "acceptable": ["none"], "reason": "r"},
            _spec(samples=5),
        )


def test_version_hash_covers_the_registry() -> None:
    edited = REGISTRY.model_dump()
    edited["tools"]["issue_refund"]["description"] = "Refund an order. Reversible."
    other = ToolSelectionEvaluator(_spec(), ToolRegistry.model_validate(edited), "same-judge-hash")
    same = ToolSelectionEvaluator(_spec(), REGISTRY, "same-judge-hash")
    assert other.version_hash() != same.version_hash()
