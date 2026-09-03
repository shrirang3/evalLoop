"""Building a suite and running it."""

from __future__ import annotations

from typing import Any

from evalloop.contracts import EvalContext, EvalResult, EvalSuite, JudgeConfig, Trace
from evalloop.evaluate import PLANNED, build_suite, run_suite

JUDGES = {"default": JudgeConfig(provider="mock", model="stub-1")}

SUITE = {
    "suite": "s",
    "evaluators": [
        {
            "id": "tool_call_match",
            "type": "json_match",
            "actual": "output.tool_calls",
            "expected": "ground_truth.tool_calls",
        },
        {
            "id": "policy_followed",
            "type": "llm_question",
            "question": "ok?",
            "ground_truth": "ground_truth.policy_followed",
        },
    ],
}


def _traces(n: int = 3) -> list[Trace]:
    return [
        Trace.model_validate(
            {
                "trace_id": f"t{i}",
                "input": {"user_request": "refund"},
                "output": {"text": "done", "tool_calls": [{"name": "issue_refund"}]},
                "ground_truth": {
                    "tool_calls": [{"name": "issue_refund"}],
                    "policy_followed": True,
                },
            }
        )
        for i in range(n)
    ]


# --- registry ---


def test_a_mixed_suite_builds() -> None:
    built = build_suite(EvalSuite.model_validate(SUITE), JUDGES)
    assert built.ok
    assert [e.id for e in built.evaluators] == ["tool_call_match", "policy_followed"]
    # Only the judged evaluator gets a judge; the deterministic one must not.
    assert set(built.judges) == {"policy_followed"}


def test_an_unimplemented_type_names_its_phase() -> None:
    """ "arrives in P2" beats a Pydantic error about a Literal, which does not
    say whether you wrote it wrong or it does not exist yet."""
    suite = EvalSuite.model_validate(
        {"suite": "s", "evaluators": [{"id": "r", "type": "regex", "actual": "output.text"}]}
    )
    built = build_suite(suite, JUDGES)
    assert not built.ok
    assert "not implemented yet" in built.errors[0]
    assert PLANNED["regex"] in built.errors[0]


def test_an_undeclared_judge_lists_the_declared_ones() -> None:
    suite = EvalSuite.model_validate(
        {
            "suite": "s",
            "evaluators": [
                {"id": "q", "type": "llm_question", "question": "ok?", "judge": "strict"}
            ],
        }
    )
    built = build_suite(suite, JUDGES)
    assert not built.ok
    assert "strict" in built.errors[0]
    assert "default" in built.errors[0]


def test_construction_errors_are_collected_not_raised() -> None:
    """A bad option on one check must not hide a missing judge on another."""
    suite = EvalSuite.model_validate(
        {
            "suite": "s",
            "evaluators": [
                {"id": "r", "type": "regex", "actual": "output.text"},
                {"id": "q", "type": "llm_question", "question": "ok?", "judge": "absent"},
            ],
        }
    )
    built = build_suite(suite, JUDGES)
    assert len(built.errors) == 2


# --- runner ---


def test_every_trace_is_evaluated_by_every_evaluator() -> None:
    built = build_suite(EvalSuite.model_validate(SUITE), JUDGES)
    summary = run_suite(_traces(3), built, run_id="r1")

    assert len(summary.results) == 6
    assert summary.traces == 3
    assert set(summary.questions) == {"tool_call_match", "policy_followed"}


def test_the_four_outcomes_are_tallied_separately() -> None:
    built = build_suite(EvalSuite.model_validate(SUITE), JUDGES)
    traces = [
        *_traces(2),
        # No ground truth: not applicable, not a failure.
        Trace.model_validate(
            {"trace_id": "t9", "input": {}, "output": {"tool_calls": []}, "ground_truth": {}}
        ),
    ]
    summary = run_suite(traces, built)

    tool = summary.questions["tool_call_match"]
    assert tool.passed == 2
    assert tool.not_applicable == 1
    assert tool.failed == 0
    assert tool.errored == 0


def test_pass_rate_excludes_not_applicable_results() -> None:
    """Counting them in the denominator would let a check that can see a tenth
    of the data report 10% and look terrible, which is not the question anyone
    is asking."""
    built = build_suite(EvalSuite.model_validate(SUITE), JUDGES)
    traces = [
        *_traces(1),
        Trace.model_validate(
            {"trace_id": "t9", "input": {}, "output": {"tool_calls": []}, "ground_truth": {}}
        ),
    ]
    tool = run_suite(traces, built).questions["tool_call_match"]

    assert tool.total == 2
    assert tool.pass_rate == 1.0  # 1 of 1 decided, not 1 of 2


def test_pass_rate_is_none_when_nothing_was_decided() -> None:
    """A holdout question with no labels has no rate, and reporting 0% would
    read as a total failure."""
    built = build_suite(EvalSuite.model_validate(SUITE), JUDGES)
    trace = Trace.model_validate(
        {"trace_id": "t9", "input": {}, "output": {"tool_calls": []}, "ground_truth": {}}
    )
    assert run_suite([trace], built).questions["tool_call_match"].pass_rate is None


def test_an_exploding_evaluator_costs_one_result_not_the_run() -> None:
    """A customer's own Python check is written by a customer. One unhandled
    KeyError in it must not take down the remaining nine thousand traces."""

    class Exploding:
        id = "boom"

        def version_hash(self) -> str:
            return "v1"

        def evaluate(self, trace: Trace, ctx: EvalContext) -> EvalResult:
            raise KeyError("customer_field")

    built = build_suite(EvalSuite.model_validate(SUITE), JUDGES)
    built.evaluators.append(Exploding())  # type: ignore[arg-type]

    summary = run_suite(_traces(2), built)
    assert summary.questions["boom"].errored == 2
    assert summary.questions["tool_call_match"].passed == 2  # unaffected
    assert len(summary.results) == 6


def test_invalid_output_is_counted_and_also_kept_out_of_pass_or_fail() -> None:
    suite = EvalSuite.model_validate(
        {
            "suite": "s",
            "evaluators": [{"id": "q", "type": "llm_question", "question": "ok?"}],
        }
    )
    built = build_suite(suite, JUDGES)
    for client in built.judges.values():
        client.provider.answers = ["not json"]  # type: ignore[attr-defined]

    question = run_suite(_traces(2), built).questions["q"]
    assert question.invalid_output == 2
    assert question.passed == 0
    assert question.failed == 0


def test_cost_and_cache_hits_roll_up() -> None:
    built = build_suite(EvalSuite.model_validate(SUITE), JUDGES)
    summary = run_suite(_traces(3), built)

    assert summary.cost_usd > 0
    assert summary.questions["tool_call_match"].cost_usd == 0.0  # no model, no cost


def test_an_empty_trace_list_still_reports_every_question() -> None:
    """Otherwise a suite that matched nothing looks like a suite that ran
    nothing, and the difference matters."""
    built = build_suite(EvalSuite.model_validate(SUITE), JUDGES)
    summary = run_suite([], built)
    assert set(summary.questions) == {"tool_call_match", "policy_followed"}
    assert all(q.total == 0 for q in summary.questions.values())


def test_deterministic_evaluators_receive_no_judge() -> None:
    """plan/001 section 3.2.1: a check the training loop cannot influence is
    only ungameable if it has no access to the judge at all."""
    captured: dict[str, Any] = {}

    class Spy:
        id = "spy"

        def version_hash(self) -> str:
            return "v1"

        def evaluate(self, trace: Trace, ctx: EvalContext) -> EvalResult:
            captured["judge"] = ctx.judge
            return EvalResult(
                trace_id=trace.trace_id, evaluator_id="spy", evaluator_version="v1", passed=True
            )

    built = build_suite(EvalSuite.model_validate(SUITE), JUDGES)
    built.evaluators.append(Spy())  # type: ignore[arg-type]
    run_suite(_traces(1), built)
    assert captured["judge"] is None
