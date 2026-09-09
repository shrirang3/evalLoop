"""Compiling failures into training rows.

Two properties carry the module. Nothing invents a target — a failure whose
correct answer is unknown is dropped and counted. And the judge's own proposal
is validated against the registry before it is trained on, so a judge that
names the right tool and parameterises it wrongly cannot teach the model its
mistake.
"""

from __future__ import annotations

from typing import Any

from evalloop.contracts import EvalResult, ToolRegistry, Trace
from evalloop.feedback import build_dpo

REGISTRY = ToolRegistry.model_validate(
    {
        "tools": {
            "issue_refund": {
                "description": "Refund an order. Irreversible.",
                "arguments": {
                    "order_id": {"type": "string", "required": True},
                    "amount": {"type": "number", "required": True},
                },
                "side_effecting": True,
            },
            "open_warranty_claim": {
                "description": "Open a replacement claim for a damaged item.",
                "arguments": {
                    "order_id": {"type": "string", "required": True},
                    "reason": {"type": "string", "enum": ["damaged", "late_request"]},
                },
                "side_effecting": True,
            },
        }
    }
)

GOOD_PROPOSAL: dict[str, Any] = {"order_id": "O1", "reason": "damaged"}


def _trace(trace_id: str = "t1", **overrides: Any) -> Trace:
    payload: dict[str, Any] = {
        "trace_id": trace_id,
        "input": {
            "system_prompt": "You are a support agent.",
            "user_request": "blender 45 days ago, arrived broken",
        },
        "output": {
            "text": "Refunded.",
            "tool_calls": [
                {"name": "issue_refund", "arguments": {"order_id": "O1", "amount": 79.0}}
            ],
        },
    }
    payload.update(overrides)
    return Trace.model_validate(payload)


def _result(
    *,
    passed: bool | None = False,
    best: str | None = "open_warranty_claim",
    arguments: dict[str, Any] | None = None,
    trace_id: str = "t1",
    **overrides: Any,
) -> EvalResult:
    raw: dict[str, Any] | None = None
    if best is not None:
        raw = {
            "best": best,
            "arguments": GOOD_PROPOSAL if arguments is None else arguments,
            "acceptable": [best],
            "reason": "outside the window",
        }
    payload: dict[str, Any] = {
        "trace_id": trace_id,
        "evaluator_id": "tool_selection",
        "evaluator_version": "ev-1",
        "passed": passed,
        "normalized_prediction": ["issue_refund"],
        "ground_truth": best,
        "raw_output": raw,
        "judge_config_hash": "judge-1",
    }
    payload.update(overrides)
    return EvalResult(**payload)


def _build(*pairs: tuple[Trace, EvalResult], **kwargs: Any) -> Any:
    return build_dpo(list(pairs), REGISTRY, **kwargs)


# --- the row ---


def test_a_failure_becomes_a_preference_pair() -> None:
    dataset = _build((_trace(), _result()))
    assert dataset.size == 1

    row = dataset.rows[0].to_json()
    assert row["chosen"]["tool_calls"] == [
        {"name": "open_warranty_claim", "arguments": GOOD_PROPOSAL}
    ]
    assert row["rejected"]["tool_calls"] == [
        {"name": "issue_refund", "arguments": {"order_id": "O1", "amount": 79.0}}
    ]


def test_every_row_carries_its_provenance() -> None:
    """Rule 7. A judge-derived row is never unlabelled."""
    row = _build((_trace(), _result())).rows[0].to_json()
    assert row["target_source"] == "judge_tool_selection"
    assert row["signal_provenance"] == "judge"
    assert row["judge_version"] == "judge-1"
    assert row["evaluator_version"] == "ev-1"
    assert row["judge_health"] == "unmeasured"


def test_the_prompt_includes_the_system_prompt() -> None:
    """Trained without it, the pair teaches the model to behave that way with
    no system prompt at all."""
    prompt = _build((_trace(), _result())).rows[0].prompt
    assert prompt[0] == {"role": "system", "content": "You are a support agent."}
    assert prompt[1]["role"] == "user"


def test_recorded_messages_are_preferred_over_the_flattened_request() -> None:
    trace = _trace(
        input={"messages": [{"role": "user", "content": "turn one"}], "user_request": "ignored"}
    )
    assert _build((trace, _result())).rows[0].prompt == [{"role": "user", "content": "turn one"}]


def test_none_compiles_to_an_empty_call_list() -> None:
    """Calling nothing is a real answer, and this is how it trains."""
    dataset = _build((_trace(), _result(best="none", arguments={})))
    assert dataset.rows[0].chosen == {"tool_calls": []}


# --- nothing is invented ---


def test_a_passing_trace_produces_no_row() -> None:
    dataset = _build((_trace(), _result(passed=True)))
    assert dataset.size == 0
    assert dataset.dropped["passed"] == 1


def test_a_not_applicable_result_produces_no_row() -> None:
    dataset = _build((_trace(), _result(passed=None)))
    assert dataset.dropped["not_applicable"] == 1


def test_an_invalid_judge_answer_produces_no_row() -> None:
    """Not a model failure. Training on it would compile an evaluation bug."""
    dataset = _build((_trace(), _result(invalid_output=True, passed=None)))
    assert dataset.dropped["judge_answered_unusably"] == 1


def test_an_evaluator_error_produces_no_row() -> None:
    dataset = _build((_trace(), _result(error="boom", passed=None)))
    assert dataset.dropped["evaluator_errored"] == 1


def test_a_failure_with_no_recorded_proposal_produces_no_row() -> None:
    dataset = _build((_trace(), _result(best=None)))
    assert dataset.dropped["no_judge_proposal_recorded"] == 1


def test_a_sealed_test_trace_is_never_compiled() -> None:
    """Rule 12, enforced by exclusion. A training row built from the sealed set
    makes every later comparison meaningless."""
    dataset = _build((_trace("t9"), _result(trace_id="t9")), sealed_trace_ids=frozenset({"t9"}))
    assert dataset.size == 0
    assert dataset.dropped["in_sealed_test_split"] == 1


# --- the judge's proposal is validated before it is trained on ---


def test_a_proposal_with_a_bad_enum_value_is_dropped() -> None:
    dataset = _build((_trace(), _result(arguments={"order_id": "O1", "reason": "broken"})))
    assert dataset.dropped["proposal_failed_argument_check"] == 1


def test_a_proposal_missing_a_required_argument_is_dropped() -> None:
    dataset = _build((_trace(), _result(arguments={"reason": "damaged"})))
    assert dataset.dropped["proposal_failed_argument_check"] == 1


def test_a_proposal_with_a_hallucinated_argument_is_dropped() -> None:
    dataset = _build(
        (_trace(), _result(arguments={"order_id": "O1", "reason": "damaged", "rush": True}))
    )
    assert dataset.dropped["proposal_failed_argument_check"] == 1


def test_a_proposal_naming_an_unregistered_tool_is_dropped() -> None:
    dataset = _build((_trace(), _result(best="refund_now", arguments={})))
    assert dataset.dropped["proposal_not_in_registry"] == 1


def test_without_a_registry_no_proposal_can_be_validated() -> None:
    """Refusing beats emitting an unchecked call: the whole point of the gate is
    that judge-derived data passes a deterministic check first."""
    dataset = build_dpo([(_trace(), _result())], None)
    assert dataset.dropped["no_registry_to_validate_proposal"] == 1


# --- the manifest ---


def test_two_builds_from_the_same_results_share_a_fingerprint() -> None:
    """What makes a dataset citable in a training run."""
    pairs = [(_trace(), _result())]
    assert build_dpo(pairs, REGISTRY).fingerprint() == build_dpo(pairs, REGISTRY).fingerprint()


def test_the_manifest_counts_every_drop_reason() -> None:
    dataset = _build(
        (_trace("t1"), _result(trace_id="t1")),
        (_trace("t2"), _result(trace_id="t2", passed=True)),
        (_trace("t3"), _result(trace_id="t3", passed=None)),
    )
    manifest = dataset.manifest()
    assert manifest["rows"] == 1
    assert manifest["dropped"] == {"not_applicable": 1, "passed": 1}
    assert manifest["dropped_total"] == 2


def test_jsonl_is_one_row_per_line() -> None:
    dataset = _build((_trace("t1"), _result(trace_id="t1")), (_trace("t2"), _result(trace_id="t2")))
    assert len(dataset.to_jsonl().strip().splitlines()) == 2
