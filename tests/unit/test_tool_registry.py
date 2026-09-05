"""The tool registry and the check built on it.

Everything here runs on traces with **no ground truth**, which is the point of
plan/002: `json_match` abstains on every production trace because no product
emits an `expected_tool_calls` column, so the deterministic floor the promotion
gate is required to have (rule 10) existed only on paper. Legality is a property
of the call and the registry, so it needs no target.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from evalloop.contracts import EvalContext, EvaluatorSpec, ToolRegistry, Trace
from evalloop.evaluate import ToolRegistryCheckEvaluator

CTX = EvalContext(run_id="r1")

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
                "preconditions": ["order.age_days <= 30"],
            },
            "open_warranty_claim": {
                "description": "Open a replacement claim for a damaged or faulty item.",
                "arguments": {
                    "order_id": {"type": "string", "required": True},
                    "reason": {
                        "type": "string",
                        "enum": ["damaged", "late_request", "faulty"],
                    },
                },
                "side_effecting": True,
            },
            "lookup_order": {
                "description": "Read-only order details.",
                "arguments": {"order_id": {"type": "string", "required": True}},
                "side_effecting": False,
            },
            "route": {"description": "Hand the conversation to another node."},
        },
        "nodes": {
            "triage": {"tools": ["lookup_order", "route"]},
            "refunds": {"tools": ["issue_refund", "open_warranty_claim", "lookup_order"]},
        },
    }
)


def _trace(calls: list[dict[str, Any]], **overrides: Any) -> Trace:
    """A trace with no ground truth at all - the shape every customer actually has."""
    payload: dict[str, Any] = {
        "trace_id": "t1",
        "input": {"user_request": "refund please"},
        "output": {"text": "Sure, refunded.", "tool_calls": calls},
    }
    payload.update(overrides)
    return Trace.model_validate(payload)


def _check(**options: Any) -> ToolRegistryCheckEvaluator:
    return ToolRegistryCheckEvaluator(
        EvaluatorSpec(
            id="tool_registry_check",
            type="tool_registry_check",
            actual="output.tool_calls",
            options=options,
        ),
        REGISTRY,
    )


# --- registry contract ---


def test_a_node_cannot_allow_an_undeclared_tool() -> None:
    with pytest.raises(ValidationError, match="not declared"):
        ToolRegistry.model_validate(
            {
                "tools": {"lookup_order": {"description": "Read-only order details."}},
                "nodes": {"triage": {"tools": ["issue_refund"]}},
            }
        )


def test_none_is_reserved_as_a_tool_name() -> None:
    """`none` is the judge's answer for "no tool should have been called".

    A tool of that name would make "no tool" and "the none tool"
    indistinguishable in every result row.
    """
    with pytest.raises(ValidationError, match="reserved"):
        ToolRegistry.model_validate({"tools": {"none": {"description": "nothing"}}})


def test_catalogue_renders_signatures_and_enums() -> None:
    catalogue = REGISTRY.catalogue("refunds")
    assert 'reason: "damaged"|"late_request"|"faulty"' in catalogue
    assert "issue_refund(order_id: string, amount: number)" in catalogue
    assert "route" not in catalogue  # not permitted at this node


def test_catalogue_is_stable_across_calls() -> None:
    """Byte-identical renders are what make the judge cache hit at all."""
    assert REGISTRY.catalogue("refunds") == REGISTRY.catalogue("refunds")


def test_choices_are_the_allowlist_plus_none() -> None:
    assert REGISTRY.choices("triage") == ["lookup_order", "route", "none"]


def test_an_unset_node_falls_back_to_every_tool() -> None:
    """A flat single-node agent degrades to global membership, not to an error."""
    assert REGISTRY.allowed(None) == frozenset(REGISTRY.tools)


def test_registry_hash_changes_when_a_description_changes() -> None:
    """A description is prompt text for the judge, so editing it is a new measurement."""
    edited = REGISTRY.model_dump()
    edited["tools"]["issue_refund"]["description"] = "Refund an order. Reversible."
    assert ToolRegistry.model_validate(edited).registry_hash() != REGISTRY.registry_hash()


# --- the check, with no ground truth anywhere ---


def test_a_legal_call_passes_without_ground_truth() -> None:
    result = _check().evaluate(
        _trace([{"name": "issue_refund", "arguments": {"order_id": "O1", "amount": 79.99}}]), CTX
    )
    assert result.passed is True
    assert result.ground_truth is None


def test_a_hallucinated_tool_fails_without_ground_truth() -> None:
    """The case that returns `passed=None` from json_match on the same trace."""
    result = _check().evaluate(
        _trace([{"name": "refund_order_now", "arguments": {"order_id": "O1"}}]), CTX
    )
    assert result.passed is False
    assert "not a registered tool" in (result.explanation or "")


def test_a_tool_used_outside_its_node_fails() -> None:
    result = _check().evaluate(
        _trace(
            [
                {
                    "name": "issue_refund",
                    "arguments": {"order_id": "O1", "amount": 5.0},
                    "node": "triage",
                }
            ]
        ),
        CTX,
    )
    assert result.passed is False
    assert "not permitted at node 'triage'" in (result.explanation or "")


def test_the_same_tool_at_its_own_node_passes() -> None:
    result = _check().evaluate(
        _trace(
            [
                {
                    "name": "issue_refund",
                    "arguments": {"order_id": "O1", "amount": 5.0},
                    "node": "refunds",
                }
            ]
        ),
        CTX,
    )
    assert result.passed is True


def test_an_unknown_node_is_reported_not_silently_permitted() -> None:
    result = _check().evaluate(
        _trace([{"name": "lookup_order", "arguments": {"order_id": "O1"}, "node": "billing"}]),
        CTX,
    )
    assert result.passed is False
    assert "not in the registry" in (result.explanation or "")


def test_a_trace_level_node_applies_to_calls_that_carry_none() -> None:
    result = _check(node_path="metadata.node").evaluate(
        _trace(
            [{"name": "issue_refund", "arguments": {"order_id": "O1", "amount": 5.0}}],
            metadata={"node": "triage"},
        ),
        CTX,
    )
    assert result.passed is False
    assert "not permitted at node 'triage'" in (result.explanation or "")


def test_missing_required_argument_fails() -> None:
    result = _check().evaluate(
        _trace([{"name": "issue_refund", "arguments": {"order_id": "O1"}}]), CTX
    )
    assert result.passed is False
    assert "missing required argument 'amount'" in (result.explanation or "")


def test_a_value_outside_an_enum_fails() -> None:
    result = _check().evaluate(
        _trace(
            [{"name": "open_warranty_claim", "arguments": {"order_id": "O1", "reason": "broken"}}]
        ),
        CTX,
    )
    assert result.passed is False
    assert "not one of" in (result.explanation or "")


def test_an_undeclared_argument_fails() -> None:
    result = _check().evaluate(
        _trace([{"name": "lookup_order", "arguments": {"order_id": "O1", "tracking": True}}]), CTX
    )
    assert result.passed is False
    assert "unexpected argument 'tracking'" in (result.explanation or "")


def test_a_boolean_is_not_a_number() -> None:
    """`True` where a number belongs is a wrong call, not a coercible one."""
    result = _check().evaluate(
        _trace([{"name": "issue_refund", "arguments": {"order_id": "O1", "amount": True}}]), CTX
    )
    assert result.passed is False
    assert "should be number" in (result.explanation or "")


def test_a_repeated_side_effecting_call_fails() -> None:
    """Two identical refunds is two refunds."""
    call = {"name": "issue_refund", "arguments": {"order_id": "O1", "amount": 79.99}}
    result = _check().evaluate(_trace([call, dict(call)]), CTX)
    assert result.passed is False
    assert "side-effecting" in (result.explanation or "")


def test_a_repeated_read_only_call_is_not_a_failure() -> None:
    """Flagging a repeated lookup would be a false failure - the expensive direction."""
    call = {"name": "lookup_order", "arguments": {"order_id": "O1"}}
    result = _check().evaluate(_trace([call, dict(call)]), CTX)
    assert result.passed is True


def test_a_repeat_of_an_undeclared_tool_is_noted_not_failed() -> None:
    """`side_effecting` unset means the author has not said. Say that, don't guess."""
    call = {"name": "route", "arguments": {}}
    result = _check().evaluate(_trace([call, dict(call)]), CTX)
    assert result.passed is True
    assert "does not declare whether" in (result.explanation or "")


def test_a_trace_with_no_tool_calls_is_not_applicable() -> None:
    """Legal, but never looked at - counting it as a pass inflates the rate."""
    result = _check().evaluate(_trace([]), CTX)
    assert result.passed is None
    assert "no tool calls" in (result.explanation or "")


def test_every_problem_in_a_trace_is_reported_not_just_the_first() -> None:
    result = _check().evaluate(
        _trace(
            [
                {"name": "refund_order_now", "arguments": {}},
                {"name": "issue_refund", "arguments": {"order_id": "O1"}},
            ]
        ),
        CTX,
    )
    assert result.passed is False
    explanation = result.explanation or ""
    assert "not a registered tool" in explanation
    assert "missing required argument" in explanation


# --- configuration refusals ---


def test_an_expected_path_is_refused() -> None:
    """Accepting one would quietly reintroduce the ground-truth dependency."""
    with pytest.raises(ValueError, match="takes no 'expected' path"):
        ToolRegistryCheckEvaluator(
            EvaluatorSpec(
                id="x",
                type="tool_registry_check",
                actual="output.tool_calls",
                expected="ground_truth.tool_calls",
            ),
            REGISTRY,
        )


def test_precondition_checking_is_refused_until_implemented() -> None:
    """Silently passing everything while appearing to enforce a policy is worse
    than refusing to be configured."""
    with pytest.raises(ValueError, match="not implemented"):
        ToolRegistryCheckEvaluator(
            EvaluatorSpec(
                id="x",
                type="tool_registry_check",
                actual="output.tool_calls",
                options={"check_preconditions": True},
            ),
            REGISTRY,
        )


def test_version_hash_covers_the_registry() -> None:
    edited = REGISTRY.model_dump()
    edited["tools"]["issue_refund"]["description"] = "Refund an order. Reversible."
    other = ToolRegistryCheckEvaluator(
        EvaluatorSpec(
            id="tool_registry_check",
            type="tool_registry_check",
            actual="output.tool_calls",
        ),
        ToolRegistry.model_validate(edited),
    )
    assert other.version_hash() != _check().version_hash()
