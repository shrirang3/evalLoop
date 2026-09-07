"""The wrong-tool report.

`plan/002` section 4 calls this table the product. Everything it needs is
already in `eval_result`: `normalized_prediction` holds what the agent called,
`ground_truth` holds the tool the judge picked without being shown that call,
and `raw_output.violations` holds the deterministic faults by code.

Two tables, deliberately not merged:

- **selection** — wrong choice among *legal* tools. Judge-derived, so relative
  claims only until the judge is calibrated (`plan/001` section 1).
- **violations** — illegal calls. Objective, no model involved, and the reason
  a promotion gate has a floor at all.
- **contradictions** — the reply describes something the calls did not do. Judged,
  but with no target either way: consistency is the answer by construction.
- **failures** — the call ran and the tool said no. The product's own verdict,
  grouped by tool and error.

Pure functions over rows. Nothing here opens a database or prints, so the
counting is testable without either.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ContradictionRow",
    "FailureRow",
    "SelectionRow",
    "ToolCallReport",
    "ViolationRow",
    "build_tool_call_report",
]

_NONE = "none"


@dataclass(frozen=True, slots=True)
class SelectionRow:
    """One (what was called, what the judge chose) pair, with its frequency."""

    called: str
    """Comma-joined tool names, or `none` when the agent called nothing."""

    judge_says: str
    count: int
    example_trace_id: str | None = None


@dataclass(frozen=True, slots=True)
class ViolationRow:
    """One class of illegal call, with its frequency."""

    code: str
    tools: tuple[str, ...]
    count: int
    example: str | None = None


@dataclass(frozen=True, slots=True)
class ContradictionRow:
    """One trace where the reply and the calls disagree."""

    trace_id: str
    called: str
    explanation: str


@dataclass(frozen=True, slots=True)
class FailureRow:
    """One (tool, error) pair that the product itself rejected."""

    tool: str
    error: str
    count: int
    example: str | None = None


@dataclass
class ToolCallReport:
    """Everything the CLI renders, already counted."""

    run_id: str
    traces: int = 0

    selection: list[SelectionRow] = field(default_factory=list)
    """Disagreements only, most frequent first. Agreement is a number, not a table."""

    selection_agreed: int = 0
    selection_skipped: int = 0
    """Not applicable - an unknown node, or no judge answer to compare."""

    selection_invalid: int = 0
    selection_evaluated: bool = False
    """False when the suite has no `tool_selection` check at all, which is a
    different statement from "it found nothing"."""

    violations: list[ViolationRow] = field(default_factory=list)
    violations_clean: int = 0
    violations_skipped: int = 0
    violations_evaluated: bool = False

    contradictions: list[ContradictionRow] = field(default_factory=list)
    """Every mismatch, listed rather than counted: unlike a tool name, no two
    contradictions are the same string, so there is nothing to group by."""

    contradictions_consistent: int = 0
    contradictions_skipped: int = 0
    contradictions_invalid: int = 0
    contradictions_evaluated: bool = False

    failures: list[FailureRow] = field(default_factory=list)
    failures_clean: int = 0
    failures_skipped: int = 0
    """Traces whose calls record no outcome at all. Usually most of them, and
    the number is the finding: it says how much of the dataset can answer the
    question (plan/003 section 2)."""

    failures_evaluated: bool = False

    @property
    def wrong_selections(self) -> int:
        return sum(row.count for row in self.selection)

    @property
    def illegal_calls(self) -> int:
        return sum(row.count for row in self.violations)

    @property
    def contradicted(self) -> int:
        return len(self.contradictions)

    @property
    def failed_calls(self) -> int:
        return sum(row.count for row in self.failures)

    @property
    def is_empty(self) -> bool:
        return not (
            self.selection_evaluated
            or self.violations_evaluated
            or self.contradictions_evaluated
            or self.failures_evaluated
        )


def build_tool_call_report(
    run_id: str,
    selection_rows: list[dict[str, Any]],
    violation_rows: list[dict[str, Any]],
    consistency_rows: list[dict[str, Any]] | None = None,
    outcome_rows: list[dict[str, Any]] | None = None,
) -> ToolCallReport:
    """Count a run's tool results into the two tables.

    Both inputs are lists of `eval_result` rows as dicts. Rows arrive already
    filtered to one evaluator id, because which evaluator carries which check is
    a suite decision and not this module's to guess.
    """
    consistency = consistency_rows or []
    outcomes = outcome_rows or []
    report = ToolCallReport(run_id=run_id)
    _count_selection(report, selection_rows)
    _count_violations(report, violation_rows)
    _count_contradictions(report, consistency)
    _count_failures(report, outcomes)
    report.traces = len(
        {row["trace_id"] for row in [*selection_rows, *violation_rows, *consistency, *outcomes]}
    )
    return report


def _count_failures(report: ToolCallReport, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    report.failures_evaluated = True

    counts: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], str] = {}

    for row in rows:
        passed = row.get("passed")
        if passed is None:
            report.failures_skipped += 1
        elif passed:
            report.failures_clean += 1

        raw = row.get("raw_output")
        found = raw.get("failures") if isinstance(raw, dict) else None
        for failure in found if isinstance(found, list) else []:
            if not isinstance(failure, dict):
                continue
            key = (str(failure.get("tool") or "—"), str(failure.get("error") or "—"))
            counts[key] += 1
            examples.setdefault(key, str(row["trace_id"]))

    report.failures = [
        FailureRow(tool=tool, error=error, count=count, example=examples[(tool, error)])
        for (tool, error), count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _count_contradictions(report: ToolCallReport, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    report.contradictions_evaluated = True

    found: list[ContradictionRow] = []
    for row in rows:
        if row.get("invalid_output"):
            report.contradictions_invalid += 1
            continue
        passed = row.get("passed")
        if passed is None:
            report.contradictions_skipped += 1
        elif passed:
            report.contradictions_consistent += 1
        else:
            found.append(
                ContradictionRow(
                    trace_id=str(row["trace_id"]),
                    called=_called(row.get("normalized_prediction")),
                    explanation=str(row.get("explanation") or ""),
                )
            )

    # Trace id, so the list is stable across runs of the same data.
    report.contradictions = sorted(found, key=lambda r: r.trace_id)


def _count_selection(report: ToolCallReport, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    report.selection_evaluated = True

    pairs: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], str] = {}

    for row in rows:
        if row.get("invalid_output"):
            report.selection_invalid += 1
            continue
        passed = row.get("passed")
        if passed is None:
            report.selection_skipped += 1
            continue
        if passed:
            report.selection_agreed += 1
            continue

        key = (_called(row.get("normalized_prediction")), _judge_choice(row.get("ground_truth")))
        pairs[key] += 1
        examples.setdefault(key, str(row["trace_id"]))

    report.selection = [
        SelectionRow(
            called=called, judge_says=judge, count=count, example_trace_id=examples[(called, judge)]
        )
        # Frequency first, then name, so two runs of the same data render the
        # same table rather than shuffling ties around.
        for (called, judge), count in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _count_violations(report: ToolCallReport, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    report.violations_evaluated = True

    counts: Counter[str] = Counter()
    tools: dict[str, set[str]] = {}
    examples: dict[str, str] = {}

    for row in rows:
        if row.get("passed") is None:
            report.violations_skipped += 1
        elif row.get("passed"):
            report.violations_clean += 1

        for violation in _violations(row.get("raw_output")):
            code = str(violation.get("code", "unknown"))
            counts[code] += 1
            if isinstance(tool := violation.get("tool"), str):
                tools.setdefault(code, set()).add(tool)
            examples.setdefault(code, str(row["trace_id"]))

    report.violations = [
        ViolationRow(
            code=code,
            tools=tuple(sorted(tools.get(code, set()))),
            count=count,
            example=examples.get(code),
        )
        for code, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def _violations(raw_output: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_output, dict):
        return []
    found = raw_output.get("violations")
    return [v for v in found if isinstance(v, dict)] if isinstance(found, list) else []


def _unwrap(stored: Any) -> Any:
    """Undo `repo._jsonable`.

    `normalized_prediction` and `ground_truth` are JSON *object* columns, so a
    scalar or a list is stored as `{"value": ...}` on the way in. Reading the
    envelope as though it were the value is silent rather than loud - every
    prediction reads as absent, and the report fills with rows saying nothing
    was called and nothing was expected.
    """
    if isinstance(stored, dict) and set(stored) == {"value"}:
        return stored["value"]
    return stored


def _called(prediction: Any) -> str:
    """`tool_selection` stores the called tool names as a list; `none` means none."""
    unwrapped = _unwrap(prediction)
    if isinstance(unwrapped, list) and unwrapped:
        return ",".join(str(name) for name in unwrapped)
    return _NONE


def _judge_choice(ground_truth: Any) -> str:
    """The judge's pick - a *computed* target, which is why it lands in the
    `ground_truth` column with a judge hash beside it (plan/002 section 8)."""
    unwrapped = _unwrap(ground_truth)
    return str(unwrapped) if isinstance(unwrapped, str) else _NONE
