"""Turn failures into training rows, and refuse to invent a target.

`plan/000` P4 states the rule this module exists to enforce: a row is emitted
only when a target comes from an allowed source, and a failure with no target is
**dropped and counted**. That number is the honest output - it says how much of
your failure set is currently unusable, and which traces are worth a human's
time first.

Of the four tool checks, exactly one produces a target for free:

    tool_selection        the judge's own pick IS the correct call
    text_matches_tools    needs a rewritten reply - generation, not yet built
    tool_registry_check   knowing `refund_order_now` does not exist says
                          nothing about what should have been called
    tool_call_outcome     the error says the call was rejected, not what
                          would have worked

The last two are objective failures and make excellent `rejected` examples and
terrible `chosen` ones. They are dropped here on purpose.

**The judge's proposal is validated before it is trained on.** A judge that
names the right tool and parameterises it wrongly would otherwise teach the
model its own mistake, so every proposed call goes through the same registry
argument check that scores production calls (`plan/002` section 8). A
deterministic check gating judge-derived data is the gate asymmetry applied one
layer earlier.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from evalloop.contracts.result import EvalResult
from evalloop.contracts.tools import NONE_CHOICE, ToolRegistry
from evalloop.contracts.trace import Trace, canonical_json

__all__ = ["Dataset", "DatasetRow", "build_dpo"]

SELECTION_SOURCE = "judge_tool_selection"
"""The seventh target source, added by plan/002 section 8 to the six in
plan/000 P4.2 and the `judge_preference_pair` of plan/001 section 5.1."""


@dataclass(frozen=True, slots=True)
class DatasetRow:
    """One preference pair, with its provenance attached rather than implied."""

    trace_id: str
    prompt: list[dict[str, str]]
    chosen: dict[str, Any]
    rejected: dict[str, Any]

    target_source: str
    signal_provenance: str
    judge_version: str | None
    evaluator_version: str
    judge_health: str
    """`unmeasured` until judge health exists. Stamped rather than blocked:
    honesty is provenance, not prohibition (plan/001 section 5.2)."""

    def to_json(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "target_source": self.target_source,
            "signal_provenance": self.signal_provenance,
            "judge_version": self.judge_version,
            "evaluator_version": self.evaluator_version,
            "judge_health": self.judge_health,
        }


@dataclass
class Dataset:
    """Rows plus the manifest, which is the part that makes them auditable."""

    rows: list[DatasetRow] = field(default_factory=list)
    dropped: Counter[str] = field(default_factory=Counter)
    """Reason -> count. A failure that produced no row is never silent."""

    @property
    def size(self) -> int:
        return len(self.rows)

    @property
    def dropped_total(self) -> int:
        return sum(self.dropped.values())

    def fingerprint(self) -> str:
        """Content hash of the emitted rows.

        Two builds from the same results produce the same fingerprint, which is
        what makes a dataset citable in a training run.
        """
        payload = canonical_json([row.to_json() for row in self.rows])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_jsonl(self) -> str:
        return "".join(f"{canonical_json(row.to_json())}\n" for row in self.rows)

    def manifest(self) -> dict[str, Any]:
        return {
            "rows": self.size,
            "fingerprint": self.fingerprint(),
            "dropped": dict(sorted(self.dropped.items())),
            "dropped_total": self.dropped_total,
        }


def build_dpo(
    pairs: Sequence[tuple[Trace, EvalResult]],
    registry: ToolRegistry | None = None,
    *,
    sealed_trace_ids: frozenset[str] = frozenset(),
) -> Dataset:
    """Compile `tool_selection` failures into preference pairs.

    Takes (trace, result) pairs because a training row needs both: the result
    carries the verdict and the judge's proposal, and the trace carries the
    prompt and the call that was actually made.
    """
    dataset = Dataset()

    for trace, result in pairs:
        reason = _reject(trace, result, sealed_trace_ids)
        if reason is not None:
            dataset.dropped[reason] += 1
            continue

        parsed = result.raw_output or {}
        best = str(parsed.get("best"))
        arguments = parsed.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}

        if best != NONE_CHOICE:
            if registry is None:
                dataset.dropped["no_registry_to_validate_proposal"] += 1
                continue
            if not registry.has(best):
                dataset.dropped["proposal_not_in_registry"] += 1
                continue
            problems = registry.check_arguments(best, arguments)
            if problems:
                # The judge named the right tool and parameterised it wrongly.
                # Training on it would teach the model the judge's mistake.
                dataset.dropped["proposal_failed_argument_check"] += 1
                continue

        dataset.rows.append(
            DatasetRow(
                trace_id=trace.trace_id,
                prompt=_prompt(trace),
                chosen=_chosen(best, arguments),
                rejected={
                    "tool_calls": [
                        call.model_dump(mode="json", include={"name", "arguments"})
                        for call in trace.output.tool_calls
                    ]
                },
                target_source=SELECTION_SOURCE,
                signal_provenance="judge",
                judge_version=result.judge_config_hash,
                evaluator_version=result.evaluator_version,
                judge_health="unmeasured",
            )
        )

    return dataset


def _reject(trace: Trace, result: EvalResult, sealed: frozenset[str]) -> str | None:
    """Why this pair cannot become a row, or None if it can."""
    if trace.trace_id in sealed:
        # Rule 12. A training row built from the sealed test set makes every
        # later comparison meaningless, so this is a build failure's-worth of
        # important even though it is counted rather than raised.
        return "in_sealed_test_split"
    if result.error is not None:
        return "evaluator_errored"
    if result.invalid_output:
        return "judge_answered_unusably"
    if result.passed is None:
        return "not_applicable"
    if result.passed:
        return "passed"
    if not isinstance(result.raw_output, dict) or "best" not in result.raw_output:
        return "no_judge_proposal_recorded"
    return None


def _prompt(trace: Trace) -> list[dict[str, str]]:
    """The prompt as the model saw it, preferring recorded messages.

    Falls back to `user_request` for products that log a turn rather than a
    conversation. The system prompt is included when present: a preference pair
    trained without it teaches the model to behave that way with no system
    prompt at all.
    """
    if trace.input.messages:
        return [{"role": m.role, "content": m.content} for m in trace.input.messages]

    messages: list[dict[str, str]] = []
    if trace.input.system_prompt:
        messages.append({"role": "system", "content": trace.input.system_prompt})
    messages.append({"role": "user", "content": trace.input.user_request or ""})
    return messages


def _chosen(best: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """`none` is a real answer, and an empty call list is how it trains."""
    if best == NONE_CHOICE:
        return {"tool_calls": []}
    return {"tool_calls": [{"name": best, "arguments": arguments}]}
