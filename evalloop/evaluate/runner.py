"""The evaluation loop.

Sequential and deliberately dull. Concurrency, budget limits, and resume are P2;
what matters at P0 is that the loop cannot be taken down by anything one
evaluator does to one trace.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from evalloop.contracts.protocols import EvalContext, Evaluator
from evalloop.contracts.result import EvalResult
from evalloop.contracts.trace import Trace
from evalloop.evaluate.base import error_result
from evalloop.evaluate.registry import BuiltSuite

__all__ = ["QuestionSummary", "RunSummary", "run_suite"]


@dataclass
class QuestionSummary:
    """Per-evaluator totals. The four counts are kept apart on purpose."""

    evaluator_id: str
    passed: int = 0
    failed: int = 0
    not_applicable: int = 0
    """Ran, but had nothing to compare against - usually no ground truth. Not a
    failure, and the size of this number is how much of the dataset the check
    cannot actually see."""

    errored: int = 0
    invalid_output: int = 0
    """The judge answered unusably. A property of the judge, not the model."""

    cost_usd: float = 0.0
    cache_hits: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.not_applicable + self.errored

    @property
    def pass_rate(self) -> float | None:
        """Over applicable results only.

        Counting not-applicable rows in the denominator would let a check that
        can only see a tenth of the data report a 10% score and look terrible,
        or report 100% of that tenth and look fine. Neither is the question
        anyone is asking.
        """
        decided = self.passed + self.failed
        return None if decided == 0 else self.passed / decided


@dataclass
class RunSummary:
    results: list[EvalResult] = field(default_factory=list)
    questions: dict[str, QuestionSummary] = field(default_factory=dict)
    traces: int = 0

    @property
    def cost_usd(self) -> float:
        return sum(q.cost_usd for q in self.questions.values())

    @property
    def cache_hits(self) -> int:
        return sum(q.cache_hits for q in self.questions.values())


def run_suite(
    traces: Sequence[Trace],
    built: BuiltSuite,
    *,
    run_id: str | None = None,
) -> RunSummary:
    """Evaluate every trace with every evaluator."""
    summary = RunSummary(traces=len(traces))
    contexts = {
        evaluator.id: EvalContext(run_id=run_id, judge=built.judges.get(evaluator.id))
        for evaluator in built.evaluators
    }

    for evaluator in built.evaluators:
        summary.questions.setdefault(evaluator.id, QuestionSummary(evaluator_id=evaluator.id))

    for trace in traces:
        for evaluator in built.evaluators:
            result = _evaluate_safely(evaluator, trace, contexts[evaluator.id])
            summary.results.append(result)
            _tally(summary.questions[evaluator.id], result)

    return summary


def _evaluate_safely(evaluator: Evaluator, trace: Trace, ctx: EvalContext) -> EvalResult:
    """The protocol says evaluators do not raise. This is the belt to that braces.

    A custom Python check is written by a customer, and one unhandled KeyError
    in it must cost that (trace, evaluator) pair rather than the remaining nine
    thousand traces.
    """
    try:
        return evaluator.evaluate(trace, ctx)
    except Exception as exc:
        return error_result(trace, evaluator.id, evaluator.version_hash(), exc)


def _tally(question: QuestionSummary, result: EvalResult) -> None:
    if result.error is not None:
        question.errored += 1
    elif result.invalid_output:
        question.invalid_output += 1
        question.errored += 1
    elif result.passed is True:
        question.passed += 1
    elif result.passed is False:
        question.failed += 1
    else:
        question.not_applicable += 1

    question.cost_usd += result.cost_usd or 0.0
    question.cache_hits += 1 if result.cache_hit else 0
