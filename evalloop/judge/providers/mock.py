"""A judge that answers from a script.

CI must be able to run the whole evaluation path with no API key, no network,
and the same answer every time. It also lets a test construct a judge that is
deliberately broken - always yes, coin flip, malformed output - which is how the
judgecard's own claims get tested in P3.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from evalloop.contracts.judgeconf import JudgeConfig
from evalloop.contracts.protocols import RenderedPrompt
from evalloop.contracts.result import TokenUsage
from evalloop.judge.client import ProviderResult

__all__ = ["MockProvider"]


@dataclass
class MockProvider:
    """Returns canned answers, in order, cycling once exhausted.

    `answer_fn` takes precedence and receives the rendered prompt, so a test can
    make the answer depend on the trace - the difference between a stub and a
    fixture that can model a biased judge.
    """

    name: str = "mock"
    answers: list[Any] | None = None
    """Canned answers, cycled. `None` means "answer whatever this schema asks
    for", which is what lets one `provider: mock` line serve a suite containing
    both an `llm_question` and a `tool_selection` - two different answer shapes,
    neither of them configured per check."""

    answer_fn: Callable[[RenderedPrompt], Any] | None = None
    raise_error: Exception | None = None
    calls: list[RenderedPrompt] = field(default_factory=list)
    tokens_in: int = 100
    tokens_out: int = 20
    cost_usd: float | None = 0.001

    def complete(
        self,
        prompt: RenderedPrompt,
        schema: dict[str, Any],
        config: JudgeConfig,
    ) -> ProviderResult:
        self.calls.append(prompt)

        if self.raise_error is not None:
            raise self.raise_error

        if self.answer_fn is not None:
            payload = self.answer_fn(prompt)
        elif self.answers is not None:
            payload = self.answers[(len(self.calls) - 1) % len(self.answers)]
        else:
            payload = _from_schema(schema)

        # A string answer is emitted verbatim, so a test can produce malformed
        # output on purpose.
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return ProviderResult(
            text=text,
            usage=TokenUsage(
                tokens_in=self.tokens_in,
                tokens_out=self.tokens_out,
                cost_usd=self.cost_usd,
            ),
        )


def _from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The dullest answer that satisfies the schema.

    Deliberately not random and not clever: a mock judge whose answers move
    between runs makes a failing CI run impossible to read. Enums pick their
    first value, which for `tool_selection` is the alphabetically first tool -
    wrong often enough to exercise the failure path, and identical every time.
    """
    answer: dict[str, Any] = {}
    for name, spec in schema.get("properties", {}).items():
        if not isinstance(spec, dict):
            continue
        if (choices := spec.get("enum")) and isinstance(choices, list) and choices:
            answer[name] = choices[0]
        elif spec.get("type") == "boolean":
            answer[name] = True
        elif spec.get("type") == "array":
            items = spec.get("items")
            inner = items.get("enum") if isinstance(items, dict) else None
            answer[name] = [inner[0]] if isinstance(inner, list) and inner else []
        elif spec.get("type") in {"number", "integer"}:
            answer[name] = 0
        else:
            answer[name] = "mock answer"
    return answer
