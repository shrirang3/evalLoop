"""The two interfaces every phase after P0 codes against.

Structural typing (`Protocol`) rather than base classes: an evaluator is
anything with the right shape, so a customer's own Python check is a first-class
evaluator without importing or subclassing anything from EvalLoop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from evalloop.contracts.result import EvalResult, JudgeResponse
from evalloop.contracts.trace import Trace

__all__ = ["EvalContext", "Evaluator", "Judge", "RenderedPrompt"]


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """A prompt after templating, immediately before it goes to a provider.

    Kept as a distinct type because two things key off the rendered form rather
    than the template: the LLM cache, and PII redaction - which runs here, after
    render and before the HTTP call, so no unredacted text can reach a third
    party.
    """

    system: str | None
    messages: list[dict[str, str]]

    def cache_key_payload(self) -> dict[str, Any]:
        return {"system": self.system, "messages": self.messages}


@runtime_checkable
class Judge(Protocol):
    """Anything that can answer a schema-constrained question about a trace."""

    def ask(self, prompt: RenderedPrompt, schema: dict[str, Any]) -> JudgeResponse:
        """Ask one question and return the raw response, the parse, and the cost.

        Implementations must not raise for a bad answer. A malformed response
        comes back as a JudgeResponse with `parsed=None`; a transport failure
        comes back with `error` set. Both are data the judgecard needs, and
        neither should take down a run of ten thousand traces.
        """
        ...


@dataclass(frozen=True, slots=True)
class EvalContext:
    """Everything an evaluator gets besides the trace itself."""

    run_id: str | None = None
    judge: Judge | None = None
    """Present only for evaluators that need one. Deterministic checks get None
    and must never acquire a judge by other means - that is what keeps their
    results ungameable by the training loop."""

    options: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Evaluator(Protocol):
    """Anything that turns a trace into a result."""

    id: str

    def version_hash(self) -> str:
        """Stable fingerprint of this evaluator's configuration.

        Derived from config, never from runtime state, so the same YAML always
        produces the same hash. Every result carries it, which is what makes
        "did this metric move, or did the check change underneath me?" an
        answerable question.
        """
        ...

    def evaluate(self, trace: Trace, ctx: EvalContext) -> EvalResult:
        """Run the check. Returns a result with `error` set rather than raising."""
        ...
