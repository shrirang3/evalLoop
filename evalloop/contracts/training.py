"""Training configuration.

Shape only at P0 - the trainer itself is P5. Frozen now so `evalloop validate`
can reject a broken `training.yaml` before anyone waits on a GPU to find out.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["LoRAConfig", "TrainingConfig"]

_STRICT = ConfigDict(extra="forbid", frozen=True)


class LoRAConfig(BaseModel):
    model_config = _STRICT

    rank: int = Field(default=16, gt=0)
    alpha: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: list[str] = Field(default_factory=list)
    quantized: bool = False


class TrainingConfig(BaseModel):
    """One `training.yaml`."""

    model_config = _STRICT

    dataset_id: str = Field(min_length=1)
    """A `feedback_dataset` row. Immutable and hashed, so a training run always
    names exactly which rows it saw."""

    backend: Literal["trl_lora"] = "trl_lora"
    strategy: Literal["sft", "dpo"] = "dpo"

    base_model: str = Field(min_length=1)
    base_revision: str | None = None
    """Pinning the revision is what makes a training run reproducible; a model
    name alone can point at different weights next month."""

    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    epochs: float = Field(default=1.0, gt=0)
    learning_rate: float = Field(default=1e-4, gt=0)
    warmup_ratio: float = Field(default=0.03, ge=0.0, lt=1.0)
    batch_size: int = Field(default=4, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    max_seq_length: int = Field(default=2048, gt=0)
    seed: int = 42

    max_truncated_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    """Refuse to launch if more than this share of examples would be cut off.
    Training on truncated targets teaches the model to stop mid-sentence."""

    min_rows: int = Field(default=50, gt=0)
    """Refuse to launch below this. A DPO run on a handful of pairs learns
    noise and produces a candidate that looks trained but is not."""

    extra: dict[str, Any] = Field(default_factory=dict)
