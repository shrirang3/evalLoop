"""Metastore schema.

Postgres holds facts *about* the data - runs, results, manifests, aggregates -
so questions like "every Hindi refund failure from last Tuesday" are a query
rather than a script. Bulk traces and results live in Parquet on the artifact
store, with pointers here.

Two tables are immutable by database trigger rather than by convention:
`snapshot` and `feedback_dataset`. Both are things later work cites as evidence.
A result from three months ago has to still mean what it said, which is only
true if nothing could have quietly edited the data underneath it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = [
    "IMMUTABLE_TABLES",
    "Base",
    "Comparison",
    "EvalResultRow",
    "EvalRun",
    "FeedbackDataset",
    "JudgeConfigRow",
    "Judgecard",
    "LLMCache",
    "ModelRegistry",
    "Project",
    "Snapshot",
    "SplitAssignment",
    "TraceRow",
    "TrainRun",
]

# JSONB on Postgres, plain JSON elsewhere, so unit tests can use SQLite without
# a running database while integration tests exercise the real thing.
Json = JSONB().with_variant(JSON(), "sqlite")

IMMUTABLE_TABLES: tuple[str, ...] = ("snapshot", "feedback_dataset")
"""Guarded by an UPDATE/DELETE trigger in migration 0001."""

SPLITS: tuple[str, ...] = ("train", "dev", "test")


class Base(DeclarativeBase):
    pass


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Project(Base):
    __tablename__ = "project"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    config_yaml: Mapped[str] = mapped_column(Text, nullable=False)
    """The project.yaml as written, verbatim. Kept so a run can be reproduced
    from the config that actually produced it rather than today's version."""
    created_at: Mapped[datetime] = _created_at()


class Snapshot(Base):
    """An immutable set of ingested traces.

    `source_fingerprint` hashes the connector config, the query, the row count,
    and the sorted content hashes. Re-ingesting identical data finds the existing
    fingerprint and returns that snapshot instead of creating a duplicate, which
    is what stops a repeated `evalloop ingest` from silently doubling a dataset.
    """

    __tablename__ = "snapshot"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("project.id", ondelete="RESTRICT"), nullable=False
    )
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    redaction_report: Mapped[dict[str, Any] | None] = mapped_column(Json, nullable=True)
    split_report: Mapped[dict[str, Any] | None] = mapped_column(Json, nullable=True)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (CheckConstraint("row_count >= 0", name="ck_snapshot_row_count"),)


class TraceRow(Base):
    """A pointer to one trace. The trace body itself is in Parquet."""

    __tablename__ = "trace"

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("snapshot.id", ondelete="CASCADE"), primary_key=True
    )
    trace_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    split: Mapped[str] = mapped_column(
        String(16), nullable=False, default="train", server_default="train"
    )
    parquet_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "split IN ('train', 'dev', 'test')",
            name="ck_trace_split",
        ),
        Index("ix_trace_snapshot_split", "snapshot_id", "split"),
        Index("ix_trace_content_hash", "content_hash"),
    )


class SplitAssignment(Base):
    """Which split a trace belongs to, and why.

    The unique constraint on (snapshot_id, trace_id) is the database-level half
    of rule 12: a trace cannot be in two splits, so it cannot be both trained on
    and tested against.
    """

    __tablename__ = "split_assignment"

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("snapshot.id", ondelete="CASCADE"), primary_key=True
    )
    trace_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    split_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    """The value that decided the assignment - a customer id, a time bucket -
    so a surprising split can be explained rather than re-derived."""
    split_strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    sealed: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    """True for the test set. Only the promotion gate may read sealed rows."""

    __table_args__ = (
        UniqueConstraint("snapshot_id", "trace_id", name="uq_split_assignment_trace"),
        CheckConstraint("split IN ('train', 'dev', 'test')", name="ck_split_assignment_split"),
    )


class JudgeConfigRow(Base):
    """A judge configuration, keyed by its own version hash.

    The hash is the primary key rather than a surrogate id: two identical
    configurations are the same judge, and one edited rubric sentence is a
    different judge with a different key. Rule 2, expressed in the schema.
    """

    __tablename__ = "judge_config"

    hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(255), nullable=False)
    params: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    rubric: Mapped[dict[str, Any] | None] = mapped_column(Json, nullable=True)
    response_schema: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False)
    created_at: Mapped[datetime] = _created_at()


class EvalRun(Base):
    __tablename__ = "eval_run"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("snapshot.id", ondelete="RESTRICT"), nullable=False
    )
    suite_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    split: Mapped[str] = mapped_column(String(16), nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    """None for a baseline run over ingested traces; set when re-evaluating a
    candidate's regenerated outputs."""

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="running", server_default="running"
    )
    cost_usd: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0, server_default=text("0")
    )
    tokens_in: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    tokens_out: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    cache_hits: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )

    started_at: Mapped[datetime] = _created_at()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'partial')",
            name="ck_eval_run_status",
        ),
        CheckConstraint("split IN ('train', 'dev', 'test')", name="ck_eval_run_split"),
        Index("ix_eval_run_snapshot", "snapshot_id", "split"),
    )


class EvalResultRow(Base):
    """One (trace, evaluator) outcome.

    `evaluator_version` is NOT NULL on purpose: the runner cannot write a result
    without recording which version of the check produced it, which is rule 3.
    Without it, "the metric moved" and "the check changed underneath me" are
    indistinguishable.
    """

    __tablename__ = "eval_result"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("eval_run.id", ondelete="CASCADE"), primary_key=True
    )
    trace_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    evaluator_id: Mapped[str] = mapped_column(String(255), primary_key=True)

    evaluator_version: Mapped[str] = mapped_column(String(64), nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    """Both nullable: a set comparison yields an F1 with no pass/fail, an exact
    match a boolean with no meaningful score. Never defaulted to 0 or False."""

    normalized_prediction: Mapped[dict[str, Any] | None] = mapped_column(Json, nullable=True)
    ground_truth: Mapped[dict[str, Any] | None] = mapped_column(Json, nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_output: Mapped[dict[str, Any] | None] = mapped_column(Json, nullable=True)

    judge_config_hash: Mapped[str | None] = mapped_column(
        ForeignKey("judge_config.hash", ondelete="RESTRICT"), nullable=True
    )
    cache_hit: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    invalid_output: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    """Nullable, never 0.0 by default: an unpriced model is a gap in the ledger,
    not a free call."""
    tokens_in: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_out: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        Index("ix_eval_result_run_evaluator", "run_id", "evaluator_id"),
        Index("ix_eval_result_run_trace", "run_id", "trace_id"),
    )


class LLMCache(Base):
    """Judge responses, keyed by judge version plus rendered prompt plus schema.

    Keying on the judge version is what makes recalibrating a prompt safe: a new
    rubric is a new hash, so it can never silently reuse answers given to the
    old question.
    """

    __tablename__ = "llm_cache"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    judge_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    response: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False)
    usage: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (Index("ix_llm_cache_created_at", "created_at"),)


class Judgecard(Base):
    """Per-question trust metrics for one run.

    `feedback build` reads eligibility from here rather than from YAML, so a
    judge that failed its checks cannot be talked into minting training data by
    editing a config file.
    """

    __tablename__ = "judgecard"

    run_id: Mapped[str] = mapped_column(
        ForeignKey("eval_run.id", ondelete="CASCADE"), primary_key=True
    )
    evaluator_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    metrics: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False)
    health: Mapped[dict[str, Any] | None] = mapped_column(Json, nullable=True)
    """Judge-health probe results. Populated without any ground truth, which is
    why it is separate from `metrics` (plan/001 section 4)."""
    eligible_for_feedback: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    created_at: Mapped[datetime] = _created_at()


class FeedbackDataset(Base):
    """An immutable compiled training set.

    The manifest records snapshot and run ids, suite and judge hashes, evaluator
    versions, filter predicates, target-source and dropped-reason histograms,
    and split fingerprints. Two builds from the same inputs produce byte
    identical output, so a training run can always name exactly which rows it
    saw and where each signal came from.
    """

    __tablename__ = "feedback_dataset"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("eval_run.id", ondelete="RESTRICT"), nullable=False
    )
    strategy: Mapped[str] = mapped_column(String(16), nullable=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint("strategy IN ('sft', 'dpo')", name="ck_feedback_dataset_strategy"),
        CheckConstraint("row_count >= 0", name="ck_feedback_dataset_row_count"),
    )


class TrainRun(Base):
    __tablename__ = "train_run"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(
        ForeignKey("feedback_dataset.id", ondelete="RESTRICT"), nullable=False
    )
    backend: Mapped[str] = mapped_column(String(32), nullable=False)
    base_model: Mapped[str] = mapped_column(String(255), nullable=False)
    base_provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    """Stored so the base-provider-vs-judge-provider check (plan/001 section
    3.1) can be re-verified from the metastore, not only from config at the time."""
    config: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False)
    manifest: Mapped[dict[str, Any] | None] = mapped_column(Json, nullable=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", server_default="pending"
    )
    artifact_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_train_run_status",
        ),
    )


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    train_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("train_run.id", ondelete="SET NULL"), nullable=True
    )
    endpoint_conf: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False)
    promoted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    """A record, not a deploy. Rule 13: nothing in this package ships a model."""
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint("kind IN ('baseline', 'candidate')", name="ck_model_registry_kind"),
    )


class Comparison(Base):
    __tablename__ = "comparison"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    baseline_model: Mapped[str] = mapped_column(String(255), nullable=False)
    candidate_model: Mapped[str] = mapped_column(String(255), nullable=False)
    run_ids: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False)
    gate_result: Mapped[dict[str, Any]] = mapped_column(Json, nullable=False)
    decision: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint("decision IN ('promote', 'reject')", name="ck_comparison_decision"),
    )
