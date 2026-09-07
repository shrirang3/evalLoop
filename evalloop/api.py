"""One constructor, one call, one report.

The CLI asks for five YAML files, a Postgres instance and four commands before
it says anything. That is the right shape for a versioned pipeline that has to
answer "did this metric move, or did the check change underneath me?" six months
later. It is the wrong shape for finding out whether the thing works on your
traces this afternoon.

    from evalloop import EvalLoop, tool_selection, registry_check

    loop = EvalLoop(
        judge="anthropic:claude-sonnet-5",
        tools="tools.yaml",
        traces="traces.jsonl",
        checks=[registry_check(), tool_selection(policy="Refunds within 30 days.")],
    )
    report = loop.run()
    report.print()

**Nothing here touches a database.** No snapshot, no run row, no migration, no
`make up`. The trade is deliberate and worth stating: results are not versioned,
so two runs are not comparable by anything the library enforces. When that
matters, the YAML and the CLI are the same engine with provenance attached.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from evalloop.contracts.judgeconf import JudgeConfig, JudgeProvider
from evalloop.contracts.suite import (
    EvalSuite,
    EvaluatorSpec,
    LLMQuestionSpec,
    SuiteEvaluator,
    TextMatchesToolsSpec,
    ToolSelectionSpec,
)
from evalloop.contracts.tools import ToolRegistry
from evalloop.contracts.trace import Trace
from evalloop.evaluate.registry import build_suite
from evalloop.evaluate.runner import RunSummary, run_suite
from evalloop.ingest.mapping import apply_mapping
from evalloop.report import ToolCallReport, build_tool_call_report, render_markdown

__all__ = [
    "EvalLoop",
    "Report",
    "llm_question",
    "registry_check",
    "text_matches_tools",
    "tool_call_outcome",
    "tool_selection",
]


# --- check factories ------------------------------------------------------
#
# Thin constructors over the same specs the YAML produces, so a check declared
# in Python and one declared in a suite file are the same object by the time
# anything runs. Ids default to the type name, which is what `report tools`
# looks for.


def registry_check(id: str = "tool_registry_check", **options: Any) -> EvaluatorSpec:
    """Is this call legal? Deterministic, needs the registry, no judge."""
    return EvaluatorSpec(
        id=id, type="tool_registry_check", actual="output.tool_calls", options=options
    )


def tool_call_outcome(id: str = "tool_call_outcome", **options: Any) -> EvaluatorSpec:
    """Did the call actually succeed? Reads the outcome the product recorded."""
    return EvaluatorSpec(
        id=id, type="tool_call_outcome", actual="output.tool_calls", options=options
    )


def tool_selection(
    policy: str | None = None,
    *,
    id: str = "tool_selection",
    judge: str = "default",
    **kwargs: Any,
) -> ToolSelectionSpec:
    """Which tool should have been called? The judge never sees the call."""
    return ToolSelectionSpec(id=id, judge=judge, policy=policy, **kwargs)


def text_matches_tools(
    *, id: str = "text_matches_tools", judge: str = "default", **kwargs: Any
) -> TextMatchesToolsSpec:
    """Does the reply describe what the calls did?"""
    return TextMatchesToolsSpec(id=id, judge=judge, **kwargs)


def llm_question(
    question: str, *, id: str, judge: str = "default", **kwargs: Any
) -> LLMQuestionSpec:
    """Any other question, asked of the judge and recorded."""
    return LLMQuestionSpec(id=id, question=question, judge=judge, **kwargs)


DEFAULT_CHECKS: tuple[str, ...] = (
    "tool_registry_check",
    "tool_call_outcome",
    "tool_selection",
    "text_matches_tools",
)
"""What you get when `checks` is left out: every tool check, which is every
check that needs no ground truth."""


# --- the result -----------------------------------------------------------


class Report:
    """What a run produced, in the shape you want to read it."""

    def __init__(self, summary: RunSummary, tools: ToolCallReport) -> None:
        self.summary = summary
        self.tools = tools

    @property
    def results(self) -> list[Any]:
        """Every (trace, evaluator) row, unaggregated."""
        return self.summary.results

    @property
    def cost_usd(self) -> float:
        return self.summary.cost_usd

    def failures(self, evaluator_id: str | None = None) -> list[Any]:
        """Rows where a check ran and said no.

        Not the same as "rows that are not passes" - a check with nothing to
        compare against, or a judge that answered unusably, is neither a pass
        nor a failure, and treating it as one is how a report starts lying.
        """
        return [
            r
            for r in self.summary.results
            if r.is_failure and (evaluator_id is None or r.evaluator_id == evaluator_id)
        ]

    def print(self, console: Console | None = None) -> None:
        from evalloop.cli.report import render

        render(console or Console(), self.tools)

    def to_markdown(self, path: str | Path | None = None) -> str:
        text = render_markdown(self.tools)
        if path is not None:
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(text, encoding="utf-8")
        return text


# --- the entry point ------------------------------------------------------


class EvalLoop:
    """Evaluate traces against tool checks, without a database.

    Every argument accepts the shape you already have: a judge as
    `"provider:model"` or a full `JudgeConfig`, tools as a path, a dict or a
    `ToolRegistry`, traces as a path, an iterable of dicts, or `Trace` objects.
    """

    def __init__(
        self,
        *,
        judge: str | JudgeConfig | dict[str, JudgeConfig],
        traces: str | Path | Iterable[dict[str, Any]] | Sequence[Trace],
        tools: str | Path | dict[str, Any] | ToolRegistry | None = None,
        checks: Sequence[SuiteEvaluator] | None = None,
        mapping: dict[str, str] | None = None,
        policy: str | None = None,
    ) -> None:
        self.judges = _judges(judge)
        self.registry = _registry(tools)
        self.traces = _traces(traces, mapping)
        self.checks = list(checks) if checks is not None else _default_checks(policy)

        built = build_suite(_suite(self.checks), self.judges, cache=None, registry=self.registry)
        if not built.ok:
            raise ValueError("; ".join(built.errors))
        self._built = built

    def run(self, limit: int | None = None) -> Report:
        traces = self.traces[:limit] if limit is not None else self.traces
        summary = run_suite(traces, self._built)
        return Report(summary, _tool_report(summary))


# --- coercion -------------------------------------------------------------


def _suite(evaluators: list[SuiteEvaluator]) -> EvalSuite:
    """Wrap checks in the model `build_suite` takes.

    The name is a placeholder and the hash it produces is never stored: both
    exist to make two runs comparable, which is a promise the database-free
    path does not make. Using the real model anyway keeps one construction path
    for evaluators rather than two that can drift.
    """
    return EvalSuite(suite="in-memory", evaluators=list(evaluators))


def _judges(judge: str | JudgeConfig | dict[str, JudgeConfig]) -> dict[str, JudgeConfig]:
    if isinstance(judge, dict):
        return judge
    if isinstance(judge, JudgeConfig):
        return {"default": judge}

    provider, _, model = judge.partition(":")
    if not model:
        raise ValueError(
            f"judge {judge!r} should be 'provider:model', e.g. 'anthropic:claude-sonnet-5'"
        )
    return {"default": JudgeConfig(provider=_provider(provider), model=model)}


def _provider(name: str) -> JudgeProvider:
    known: tuple[str, ...] = ("anthropic", "openai_compat", "mock")
    if name not in known:
        raise ValueError(f"unknown judge provider {name!r}; known: {', '.join(known)}")
    return name  # type: ignore[return-value]


def _registry(tools: str | Path | dict[str, Any] | ToolRegistry | None) -> ToolRegistry | None:
    if tools is None or isinstance(tools, ToolRegistry):
        return tools
    if isinstance(tools, dict):
        return ToolRegistry.model_validate(tools)
    return ToolRegistry.model_validate(yaml.safe_load(Path(tools).read_text(encoding="utf-8")))


def _traces(
    traces: str | Path | Iterable[dict[str, Any]] | Sequence[Trace],
    mapping: dict[str, str] | None,
) -> list[Trace]:
    if isinstance(traces, (str, Path)):
        rows: list[dict[str, Any]] = [
            json.loads(line)
            for line in Path(traces).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        listed = list(traces)
        if listed and isinstance(listed[0], Trace):
            return [t for t in listed if isinstance(t, Trace)]
        rows = [r for r in listed if isinstance(r, dict)]

    if mapping is None:
        # Already in trace shape. Validated rather than trusted, so a typo'd key
        # is an error here instead of a mystery `not applicable` later.
        return [Trace.model_validate(row) for row in rows]

    result = apply_mapping([(row, f"row:{i}") for i, row in enumerate(rows, 1)], mapping)
    if result.errors:
        raise ValueError("; ".join(result.errors[:5]))
    return result.traces


def _default_checks(policy: str | None) -> list[SuiteEvaluator]:
    return [
        registry_check(),
        tool_call_outcome(),
        tool_selection(policy=policy),
        text_matches_tools(),
    ]


def _tool_report(summary: RunSummary) -> ToolCallReport:
    """Group the flat result stream the way `evalloop report tools` does."""

    def rows(evaluator_id: str) -> list[dict[str, Any]]:
        return [
            {
                "trace_id": r.trace_id,
                "passed": r.passed,
                "normalized_prediction": r.normalized_prediction,
                "ground_truth": r.ground_truth,
                "raw_output": r.raw_output,
                "invalid_output": r.invalid_output,
                "explanation": r.explanation,
            }
            for r in summary.results
            if r.evaluator_id == evaluator_id
        ]

    return build_tool_call_report(
        "in-memory",
        rows("tool_selection"),
        rows("tool_registry_check"),
        rows("text_matches_tools"),
        rows("tool_call_outcome"),
    )
