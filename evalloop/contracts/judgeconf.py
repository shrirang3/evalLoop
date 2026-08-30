"""Judge configuration and the version hash that pins it.

Rule 2 of the non-negotiables: every judge configuration is hashed. The hash
covers the provider, the model, the sampling parameters, the prompt, the
question, and the response schema - everything that could change an answer.
Edit one sentence of a rubric and you get a new hash, a new cache namespace, and
results that can never be silently compared against the old ones.

This is what makes "did the model improve, or did I change the ruler?" an
answerable question rather than a guess.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from evalloop.contracts.trace import canonical_json

__all__ = [
    "PARSER_VERSION",
    "JudgeConfig",
    "JudgeProvider",
    "judge_version_hash",
]

JudgeProvider = Literal["anthropic", "openai_compat", "mock"]

PARSER_VERSION = "1"
"""Bumped when response parsing changes in a way that could alter a parsed
answer. Part of the hash, so a parser fix invalidates caches rather than mixing
old and new interpretations of the same raw response."""

_STRICT = ConfigDict(extra="forbid", frozen=True)


class JudgeConfig(BaseModel):
    """One named judge, as declared in `judges.yaml`.

    The API key is named, never inlined - `api_key_env` points at an environment
    variable. A secret in a config file ends up in git, in the metastore, and in
    the experiment bundle.
    """

    model_config = _STRICT

    provider: JudgeProvider
    model: str = Field(min_length=1)

    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    """Defaults to 0. A judge is an instrument; sampling noise in an instrument
    shows up later as unexplained variance in the judgecard."""

    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int = Field(default=1024, gt=0)

    base_url: str | None = None
    """Required by `openai_compat` for anything that is not OpenAI itself -
    vLLM, Together, Groq, a LiteLLM proxy."""

    api_key_env: str | None = None
    timeout_s: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0)
    """Retries apply to 429 and 5xx only. Timeouts are never retried silently;
    they count toward the invalid rate, because a judge that times out on a
    third of calls is a fact the judgecard needs to report."""

    def identity(self) -> dict[str, Any]:
        """The parts of this config that can change an answer.

        `base_url`, `timeout_s`, and `max_retries` are deliberately excluded:
        pointing at a different replica of the same model does not change what
        the model says, and including them would fragment the cache for no
        reason.
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }


def judge_version_hash(
    config: JudgeConfig,
    *,
    system_prompt: str | None,
    questions: list[str],
    response_schema: dict[str, Any],
) -> str:
    """Stable fingerprint of a judge *as applied to a specific question*.

    Both halves matter. The same model asked a different question is a different
    instrument, and the same question asked of a different model is too. Every
    EvalResult carries this hash, and the LLM cache is keyed by it.
    """
    payload = {
        **config.identity(),
        "system_prompt": system_prompt,
        "questions": questions,
        "response_schema": response_schema,
        "parser_version": PARSER_VERSION,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
