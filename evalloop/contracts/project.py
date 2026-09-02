"""Project configuration: where traces come from, how they map, and the
integrity rules the rest of the pipeline is held to.

`models` and `integrity` are the config surface of plan/001 section 3. They live
here rather than in `training.yaml` because they are project-level policy - true
of every run, not of one training job - and because the checks they enable are
cross-file: base model comes from here, the judge from `judges.yaml`, and the
violation has to be caught before anything expensive starts.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evalloop.contracts.judgeconf import JudgeConfig
from evalloop.contracts.paths import split_path

ProviderRole = Literal["base", "judge"]
_DEFAULT_PROVIDER_ROLES: list[ProviderRole] = ["base", "judge"]

__all__ = [
    "BaseModelSpec",
    "GateIntegrity",
    "IntegrityConfig",
    "ProjectConfig",
    "RedactionRule",
    "SourceConfig",
    "SplitConfig",
    "check_integrity",
]

_STRICT = ConfigDict(extra="forbid", frozen=True)


class SourceConfig(BaseModel):
    """Where the traces come from. Read-only, always."""

    model_config = _STRICT

    type: Literal["jsonl", "csv", "postgres", "pyiter"]

    path: str | None = None
    """For `jsonl` and `csv`."""

    url: str | None = None
    query: str | None = None
    """For `postgres`. The query is checked to be SELECT/WITH-only before it
    runs (P1); a connector that can write is a connector that can corrupt a
    customer's production database."""

    callable: str | None = None
    """For `pyiter`: `module:function` yielding dicts."""

    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _required_field_for_type(self) -> SourceConfig:
        required: dict[str, str] = {
            "jsonl": "path",
            "csv": "path",
            "postgres": "query",
            "pyiter": "callable",
        }
        field = required[self.type]
        if getattr(self, field) is None:
            raise ValueError(f"source.type '{self.type}' requires source.{field}")
        return self


class RedactionRule(BaseModel):
    """PII removal. Applied at ingest and again before any external judge call."""

    model_config = _STRICT

    type: Literal["regex", "named_entity", "field_drop", "hash", "python"]
    fields: list[str] = Field(default_factory=list)
    pattern: str | None = None
    replacement: str = "[REDACTED]"
    entity: str | None = None
    callable: str | None = None


class SplitConfig(BaseModel):
    """How traces are divided into train / dev / test. Engine lands in P1."""

    model_config = _STRICT

    strategy: Literal["by_field", "by_time", "hash_of_field", "stratified_by_field", "random"] = (
        "hash_of_field"
    )
    field: str | None = None
    ratios: dict[str, float] = Field(
        default_factory=lambda: {"train": 0.7, "dev": 0.15, "test": 0.15}
    )
    seed: int = 42

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> SplitConfig:
        total = sum(self.ratios.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"splits.ratios must sum to 1.0, got {total}")
        if "test" not in self.ratios:
            raise ValueError("splits.ratios must include 'test' - the sealed set is not optional")
        return self


class BaseModelSpec(BaseModel):
    """The model being served and fine-tuned. Not the judge."""

    model_config = _STRICT

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    revision: str | None = None


class GateIntegrity(BaseModel):
    """The three mechanisms that stop a judge from grading its own homework.

    A judge that mints training pairs and then scores the promotion gate will
    pass the candidate by construction: it was optimised to please that grader.
    The failure is silent and inverted - judge scores rise while quality falls.
    Each field below puts something in the gate that the training loop could not
    influence (plan/001 section 3.2).
    """

    model_config = _STRICT

    holdout_questions: list[str] = Field(default_factory=list)
    """Evaluator ids used only at the gate, never compiled into training data."""

    deterministic_required: bool = True
    """The gate must contain at least one check with no model in it."""

    block_on_divergence: bool = True
    """Reject when judge scores rise while deterministic pass rate falls. That
    pattern is the signature of a candidate that learned to please the judge."""


class IntegrityConfig(BaseModel):
    """Project-level rules, enforced at `evalloop validate` rather than at run time."""

    model_config = _STRICT

    require_distinct_providers: list[ProviderRole] = Field(
        default_factory=lambda: list(_DEFAULT_PROVIDER_ROLES)
    )
    """Judges measurably prefer outputs from their own model family. Sharing a
    provider between the model under test and the model grading it builds that
    bias straight into every number."""

    gate: GateIntegrity = Field(default_factory=GateIntegrity)


class ProjectConfig(BaseModel):
    """One `project.yaml`."""

    model_config = _STRICT

    name: str = Field(min_length=1)
    description: str | None = None

    source: SourceConfig
    mapping: dict[str, str] = Field(default_factory=dict)
    """Trace path -> source field. `input.user_request: user_transcript` reads
    as "our `input.user_request` comes from their `user_transcript` column"."""

    redaction: list[RedactionRule] = Field(default_factory=list)
    splits: SplitConfig = Field(default_factory=SplitConfig)

    models: dict[Literal["base"], BaseModelSpec] | None = None
    integrity: IntegrityConfig = Field(default_factory=IntegrityConfig)

    @model_validator(mode="after")
    def _mapping_targets_are_wellformed(self) -> ProjectConfig:
        """Reject a malformed target path at validate time, not at ingest.

        `input..user_request` is a typo, and finding out about it after a
        connection has been opened and half a file read is strictly worse than
        finding out from `evalloop validate`.
        """
        for target in self.mapping:
            try:
                split_path(target)
            except ValueError as exc:
                raise ValueError(f"mapping target {target!r} is not a valid path: {exc}") from exc
        return self

    @model_validator(mode="after")
    def _trace_id_is_mapped(self) -> ProjectConfig:
        if self.mapping and "trace_id" not in self.mapping:
            raise ValueError(
                "mapping must include 'trace_id' - without a stable id, results "
                "cannot be joined back to the trace that produced them"
            )
        return self


def check_integrity(
    project: ProjectConfig,
    judges: dict[str, JudgeConfig],
) -> list[str]:
    """Cross-file integrity checks. Returns human-readable violations, empty if clean.

    Separate from Pydantic validation because it spans two files: the base model
    is declared in `project.yaml` and the judges in `judges.yaml`, and neither
    can see the other on its own.
    """
    violations: list[str] = []

    wants_distinct = {"base", "judge"} <= set(project.integrity.require_distinct_providers)
    if wants_distinct and project.models is not None and (base := project.models.get("base")):
        for name, judge in judges.items():
            if judge.provider == base.provider:
                violations.append(
                    f"integrity.require_distinct_providers: base model and judge "
                    f"'{name}' both use provider '{base.provider}'. A judge favours "
                    f"its own family's outputs, so this bias lands in every score. "
                    f"Use a different provider for one of them."
                )

    for question_id in project.integrity.gate.holdout_questions:
        if not question_id.strip():
            violations.append("integrity.gate.holdout_questions contains an empty id")

    return violations
