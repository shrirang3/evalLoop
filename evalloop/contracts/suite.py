"""The evaluation suite - what gets checked, and how.

One suite file declares every check that runs against a snapshot. Two families
live here and the difference between them is the safety property the whole
trusted-judge design rests on (plan/001 section 3.2):

- **Deterministic** checks compare values. No model, no cost, no opinion. A
  training loop cannot optimise its way past a JSON parser, which is why every
  promotion gate is required to contain at least one.
- **LLM question** checks ask a judge. Cheap to write, applicable to things no
  matcher can express, and capable of being confidently wrong - which is what
  the judgecard exists to measure.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evalloop.contracts.trace import canonical_json

__all__ = [
    "DETERMINISTIC_TYPES",
    "EvalSuite",
    "EvaluatorSpec",
    "LLMQuestionSpec",
    "MatcherType",
    "SuiteEvaluator",
    "ToolSelectionSpec",
]

DeterministicType = Literal[
    "exact_match",
    "json_match",
    "tool_registry_check",
    "json_schema",
    "regex",
    "numeric_tolerance",
    "set_comparison",
    "tool_call_exec",
    "python",
]

DETERMINISTIC_TYPES: frozenset[str] = frozenset(
    (
        "exact_match",
        "json_match",
        "tool_registry_check",
        "json_schema",
        "regex",
        "numeric_tolerance",
        "set_comparison",
        "tool_call_exec",
        "python",
    )
)

MatcherType = Literal["exact_label", "label_map", "numeric_tolerance", "boolean", "set_overlap"]

_STRICT = ConfigDict(extra="forbid", frozen=True)


class EvaluatorSpec(BaseModel):
    """A deterministic check: pull two values out of the trace and compare them."""

    model_config = _STRICT

    id: str = Field(min_length=1)
    type: DeterministicType

    actual: str
    """Dotted path to what the model produced, e.g. `output.tool_calls`."""

    expected: str | None = None
    """Dotted path to what it should have been, usually under `ground_truth`.
    Optional: `json_schema` and `regex` validate a shape rather than compare
    against a target, and work fine on traces with no ground truth at all."""

    options: dict[str, Any] = Field(default_factory=dict)
    """Type-specific knobs - `ignore_order`, `abs_tol`, `must_match`. Validated
    by each evaluator when it is constructed (P0.7/P2), not here, so this file
    does not have to know every matcher's vocabulary."""

    weight: float = Field(default=1.0, ge=0.0)
    holdout: bool = False
    """Reserved for the promotion gate; never compiled into training data.
    See `LLMQuestionSpec.holdout` - the reasoning is the same."""

    def version_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "actual": self.actual,
            "expected": self.expected,
            "options": self.options,
        }


class LLMQuestionSpec(BaseModel):
    """A question put to a judge, with a schema-constrained answer."""

    model_config = _STRICT

    id: str = Field(min_length=1)
    type: Literal["llm_question"] = "llm_question"

    judge: str = "default"
    """Name of a judge declared in `judges.yaml`."""

    question: str = Field(min_length=1)
    system_prompt: str | None = None

    inputs: dict[str, str] = Field(default_factory=dict)
    """Template variable -> dotted path. `{transcript: output.text}` makes
    `{{ transcript }}` available in the question."""

    response_schema: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {"answer": {"type": "boolean"}, "reason": {"type": "string"}},
            "required": ["answer"],
        }
    )

    matcher: MatcherType = "boolean"
    matcher_options: dict[str, Any] = Field(default_factory=dict)

    ground_truth: str | None = None
    """Dotted path to a human label for this question, when one exists.

    Optional by design. Without it the question is still measured and still
    reported - it just carries `Calibrated against GT: No` on the judgecard and
    cannot support an absolute claim (plan/001 section 1)."""

    holdout: bool = False
    """When true this question is used only at the promotion gate and is never
    compiled into training data.

    This is one of the three mechanisms that break circular measurement
    (plan/001 section 3.2.2): the candidate was never optimised against this
    rubric, so the judge's opinion of it still carries information even though
    the same judge minted the training pairs."""

    weight: float = Field(default=1.0, ge=0.0)

    def version_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "judge": self.judge,
            "question": self.question,
            "system_prompt": self.system_prompt,
            "inputs": self.inputs,
            "response_schema": self.response_schema,
            "matcher": self.matcher,
            "matcher_options": self.matcher_options,
        }


class ToolSelectionSpec(BaseModel):
    """Ask a judge which tool *should* have been called, without showing it the call.

    The other LLM check grades a decision it has been shown. This one makes the
    decision first, blind, and lets code compare afterwards - which is the same
    rule `plan/001` section 6.2 imposes on human labellers, for the same reason:
    a judge shown a verdict rationalises it (`plan/002` section 3).

    There is no `question` field. The prompt is generated from the registry, so
    the answer space is closed: the judge picks from this node's allowlist plus
    `none`, and a hallucinated answer is unrepresentable rather than merely
    unlikely.
    """

    model_config = _STRICT

    id: str = Field(min_length=1)
    type: Literal["tool_selection"] = "tool_selection"

    judge: str = "default"

    actual: str = "output.tool_calls"
    """The calls that happened. Read *after* the judge has answered, never shown to it."""

    request: str = "input.user_request"
    """What the user asked. This, the catalogue, and `policy` are the whole prompt."""

    policy: str | None = None
    """Rules the judge should apply, e.g. the refund window. Free text, hashed
    into the evaluator version because it changes the measurement."""

    inputs: dict[str, str] = Field(default_factory=dict)
    """Extra template variables, same path syntax as `LLMQuestionSpec.inputs`.
    Anything naming `output.text` or `output.tool_calls` is refused: those leak
    the decision the judge is being asked to make."""

    node_path: str | None = None
    """Dotted path to a trace-level node, for products that record one per turn
    rather than per call."""

    samples: int = 1
    """How many times to ask. Values above 1 need the sampling and cache
    machinery that arrives with the P3a self-consistency probe, and are refused
    rather than silently collapsed into one answer."""

    system_prompt: str | None = None
    holdout: bool = False
    weight: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def _inputs_do_not_leak_the_answer(self) -> ToolSelectionSpec:
        leaks = sorted(
            name
            for name, path in self.inputs.items()
            if path == "output.tool_calls" or path.startswith(("output.tool_calls", "output.text"))
        )
        if leaks:
            raise ValueError(
                f"inputs {leaks} read from the model's own output. A judge shown the "
                "call it is assessing rationalises it - that is the anchoring failure "
                "select-mode exists to avoid (plan/002 section 3)"
            )
        return self

    def version_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "judge": self.judge,
            "request": self.request,
            "policy": self.policy,
            "inputs": self.inputs,
            "system_prompt": self.system_prompt,
            "samples": self.samples,
        }


SuiteEvaluator = Annotated[
    EvaluatorSpec | LLMQuestionSpec | ToolSelectionSpec,
    Field(discriminator="type"),
]
"""Discriminated on `type`, not a plain union.

An undiscriminated union makes Pydantic try every member and report the failures
of all of them, so one typo'd key yields six errors - and the branch name lands
in the error path, where it has no corresponding line in the YAML. Discriminating
means one mistake produces one error, pointing at the token that caused it."""


class EvalSuite(BaseModel):
    """Every check that runs against one snapshot."""

    model_config = _STRICT

    suite: str = Field(min_length=1)
    description: str | None = None
    evaluators: list[SuiteEvaluator] = Field(min_length=1)

    @model_validator(mode="after")
    def _ids_are_unique(self) -> EvalSuite:
        seen: set[str] = set()
        for spec in self.evaluators:
            if spec.id in seen:
                raise ValueError(
                    f"duplicate evaluator id {spec.id!r}; ids key every result row "
                    "and must be unique within a suite"
                )
            seen.add(spec.id)
        return self

    @property
    def holdout_ids(self) -> list[str]:
        """Evaluators reserved for the gate. P4 asserts none of these reach a dataset."""
        return [e.id for e in self.evaluators if e.holdout]

    @property
    def deterministic_ids(self) -> list[str]:
        """Checks a judge cannot influence. P6 requires the gate to contain at least one."""
        return [e.id for e in self.evaluators if isinstance(e, EvaluatorSpec)]

    def suite_hash(self) -> str:
        """Fingerprint of the whole suite, stored on every run.

        Two runs with the same suite hash are comparable. Two runs without it
        are not, however similar the YAML looks.
        """
        payload = {
            "suite": self.suite,
            "evaluators": [e.version_payload() for e in self.evaluators],
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
