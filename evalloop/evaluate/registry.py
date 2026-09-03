"""Turn suite specs into evaluator objects.

A registry rather than a chain of `if type == ...`, so adding a check means
adding a class and one dict entry. Unimplemented types are refused by name and
phase - "arrives in P2" beats a Pydantic error about a Literal, which tells the
reader nothing about whether they wrote it wrong or it does not exist yet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from evalloop.contracts.judgeconf import JudgeConfig
from evalloop.contracts.protocols import Evaluator
from evalloop.contracts.suite import EvalSuite, EvaluatorSpec, LLMQuestionSpec
from evalloop.evaluate.deterministic.exact import ExactMatchEvaluator
from evalloop.evaluate.deterministic.json_match import JsonMatchEvaluator
from evalloop.evaluate.llm.question import LLMQuestionEvaluator
from evalloop.judge import make_provider
from evalloop.judge.client import CacheBackend, JudgeClient

__all__ = ["DETERMINISTIC", "PLANNED", "BuiltSuite", "build_suite"]

DETERMINISTIC: dict[str, Callable[[EvaluatorSpec], Evaluator]] = {
    "exact_match": ExactMatchEvaluator,
    "json_match": JsonMatchEvaluator,
}

PLANNED: dict[str, str] = {
    "json_schema": "P2",
    "regex": "P2",
    "numeric_tolerance": "P2",
    "set_comparison": "P2",
    "tool_call_exec": "P2",
    "python": "P2",
}


@dataclass
class BuiltSuite:
    """Evaluators ready to run, plus the judge each one talks to."""

    evaluators: list[Evaluator] = field(default_factory=list)
    judges: dict[str, JudgeClient] = field(default_factory=dict)
    """Keyed by evaluator id. Different questions may use different judges, so
    the runner builds a context per evaluator rather than one per run."""

    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def build_suite(
    suite: EvalSuite,
    judges: dict[str, JudgeConfig],
    *,
    cache: CacheBackend | None = None,
) -> BuiltSuite:
    """Construct every evaluator in a suite, collecting construction errors.

    Errors are collected rather than raised so a suite with three problems
    reports three, and so a bad option on one check does not hide a missing
    judge on another.
    """
    built = BuiltSuite()

    for spec in suite.evaluators:
        try:
            if isinstance(spec, LLMQuestionSpec):
                _add_llm(built, spec, judges, cache)
            else:
                _add_deterministic(built, spec)
        except (ValueError, KeyError, TypeError) as exc:
            built.errors.append(f"{spec.id}: {exc}")

    return built


def _add_deterministic(built: BuiltSuite, spec: EvaluatorSpec) -> None:
    factory = DETERMINISTIC.get(spec.type)
    if factory is None:
        phase = PLANNED.get(spec.type, "a later phase")
        raise ValueError(
            f"evaluator type '{spec.type}' is not implemented yet (arrives in {phase}); "
            f"available now: {', '.join(sorted(DETERMINISTIC))}"
        )
    built.evaluators.append(factory(spec))


def _add_llm(
    built: BuiltSuite,
    spec: LLMQuestionSpec,
    judges: dict[str, JudgeConfig],
    cache: CacheBackend | None,
) -> None:
    config = judges.get(spec.judge)
    if config is None:
        raise KeyError(
            f"judge '{spec.judge}' is not declared in judges.yaml; "
            f"declared: {', '.join(sorted(judges)) or '(none)'}"
        )

    client = JudgeClient(
        config,
        make_provider(config.provider),
        system_prompt=spec.system_prompt,
        questions=[spec.question],
        response_schema=spec.response_schema,
        cache=cache,
    )
    built.judges[spec.id] = client
    built.evaluators.append(LLMQuestionEvaluator(spec, client.version_hash))
