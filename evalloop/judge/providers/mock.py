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
    answers: list[Any] = field(default_factory=lambda: [{"answer": True}])
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
        else:
            payload = self.answers[(len(self.calls) - 1) % len(self.answers)]

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
