"""The judge client: retries, repair, and the boundary that never raises.

One rule governs this module. A run of ten thousand traces must not die because
one call to a provider went wrong, so `ask` always returns a JudgeResponse.
Three outcomes stay distinct in that response because they mean different
things, and conflating them is how a broken judge comes to look like a bad
model:

    parsed set          the judge answered and the answer was usable
    parsed None         the judge answered and the answer was not usable
    error set           the call never produced an answer

The judgecard reports the second and third as separate rates. A judge returning
malformed JSON on a third of calls is broken; a judge behind a flaky network is
not, and the fix is different in each case.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from evalloop.contracts.judgeconf import JudgeConfig, judge_version_hash
from evalloop.contracts.protocols import RenderedPrompt
from evalloop.contracts.result import JudgeResponse, TokenUsage
from evalloop.contracts.trace import canonical_json

__all__ = [
    "JudgeClient",
    "Provider",
    "ProviderError",
    "ProviderResult",
    "RateLimited",
    "ServerError",
    "Timeout",
    "cache_key",
]

_REPAIR_INSTRUCTION = (
    "Your previous reply could not be parsed as JSON matching the required schema. "
    "Reply again with only the JSON object, no prose and no code fences.\n"
    "The parse error was: {error}"
)


class ProviderError(Exception):
    """A call failed in a way that is not worth retrying."""


class RateLimited(ProviderError):
    """429. Worth retrying with backoff."""


class ServerError(ProviderError):
    """5xx. Worth retrying with backoff."""


class Timeout(ProviderError):
    """Deliberately not retried.

    A judge that times out on a third of calls is a property of that judge, and
    the judgecard reports it. Retrying silently would hide the finding and
    triple the bill.
    """


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """What a provider returns: the text, and what it cost."""

    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)


class Provider(Protocol):
    """One vendor's HTTP shape, behind one interface."""

    name: str

    def complete(
        self,
        prompt: RenderedPrompt,
        schema: dict[str, Any],
        config: JudgeConfig,
    ) -> ProviderResult:
        """Send one request. Raise a ProviderError subclass on failure."""
        ...


class CacheBackend(Protocol):
    """Somewhere to remember answers. Optional."""

    def get(self, key: str) -> tuple[dict[str, Any], dict[str, Any]] | None: ...

    def put(
        self,
        key: str,
        *,
        judge_config_hash: str,
        response: dict[str, Any],
        usage: dict[str, Any],
    ) -> None: ...


def cache_key(
    version_hash: str,
    prompt: RenderedPrompt,
    schema: dict[str, Any],
) -> str:
    """Cache identity: the judge version, the rendered prompt, and the schema.

    Keyed by judge *version*, which is what makes recalibrating a prompt safe.
    A new rubric is a new hash, so it can never silently reuse answers given to
    the old question - the single most important property of this cache.
    """
    payload = {
        "judge": version_hash,
        "prompt": prompt.cache_key_payload(),
        "schema": schema,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class JudgeClient:
    """A judge bound to one question.

    Constructed per evaluator rather than per run, because the version hash
    covers the question and the schema as well as the model. The same model
    asked a different question is a different instrument.
    """

    def __init__(
        self,
        config: JudgeConfig,
        provider: Provider,
        *,
        system_prompt: str | None,
        questions: list[str],
        response_schema: dict[str, Any],
        cache: CacheBackend | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.config = config
        self.provider = provider
        self.system_prompt = system_prompt
        self.questions = questions
        self.response_schema = response_schema
        self.cache = cache
        self._sleep = sleep

        self.version_hash = judge_version_hash(
            config,
            system_prompt=system_prompt,
            questions=questions,
            response_schema=response_schema,
        )

    def ask(self, prompt: RenderedPrompt, schema: dict[str, Any]) -> JudgeResponse:
        """Ask, and return what happened. Never raises."""
        key = cache_key(self.version_hash, prompt, schema)

        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            cached, _ = hit
            return JudgeResponse(
                raw=str(cached.get("raw", "")),
                parsed=cached.get("parsed"),
                # A cache hit spent no tokens and no money. Replaying the
                # original usage would bill the same call twice and make a
                # fully-cached run look as expensive as the first one - so
                # cost is a known zero here, not an unknown None.
                usage=TokenUsage(tokens_in=0, tokens_out=0, cost_usd=0.0),
                judge_config_hash=self.version_hash,
                cache_hit=True,
                latency_ms=0,
            )

        started = time.monotonic()
        try:
            result = self._call_with_retries(prompt, schema)
        except ProviderError as exc:
            return JudgeResponse(
                raw="",
                parsed=None,
                judge_config_hash=self.version_hash,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=_elapsed_ms(started),
            )

        parsed, parse_error = _parse(result.text)

        # One repair attempt. Models that wrap JSON in prose usually comply when
        # told exactly what went wrong; a second failure is a property of the
        # judge and belongs in the invalid-output rate, not in another retry.
        if parsed is None:
            repaired = self._attempt_repair(prompt, schema, result, parse_error)
            if repaired is not None:
                result, parsed = repaired

        judged = JudgeResponse(
            raw=result.text,
            parsed=parsed,
            usage=result.usage,
            judge_config_hash=self.version_hash,
            latency_ms=_elapsed_ms(started),
        )

        if self.cache is not None and parsed is not None:
            self.cache.put(
                key,
                judge_config_hash=self.version_hash,
                response={"raw": judged.raw, "parsed": parsed},
                usage=judged.usage.model_dump(),
            )
        return judged

    def _attempt_repair(
        self,
        prompt: RenderedPrompt,
        schema: dict[str, Any],
        first: ProviderResult,
        parse_error: str,
    ) -> tuple[ProviderResult, dict[str, Any]] | None:
        repair_prompt = RenderedPrompt(
            system=prompt.system,
            messages=[
                *prompt.messages,
                {"role": "assistant", "content": first.text},
                {"role": "user", "content": _REPAIR_INSTRUCTION.format(error=parse_error)},
            ],
        )
        try:
            second = self._call_with_retries(repair_prompt, schema)
        except ProviderError:
            return None

        parsed, _ = _parse(second.text)
        if parsed is None:
            return None

        # Both calls were paid for, so both are billed.
        combined = ProviderResult(
            text=second.text,
            usage=TokenUsage(
                tokens_in=first.usage.tokens_in + second.usage.tokens_in,
                tokens_out=first.usage.tokens_out + second.usage.tokens_out,
                cost_usd=_add_costs(first.usage.cost_usd, second.usage.cost_usd),
            ),
        )
        return combined, parsed

    def _call_with_retries(
        self,
        prompt: RenderedPrompt,
        schema: dict[str, Any],
    ) -> ProviderResult:
        last: ProviderError | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                return self.provider.complete(prompt, schema, self.config)
            except (RateLimited, ServerError) as exc:
                last = exc
                if attempt < self.config.max_retries:
                    self._sleep(_backoff(attempt))
            except ProviderError:
                # Timeout and everything else: surfaced, not retried.
                raise
        assert last is not None
        raise last


def _parse(text: str) -> tuple[dict[str, Any] | None, str]:
    """Best-effort JSON extraction.

    Tolerates a code fence and surrounding prose, because a judge that is right
    about the answer and wrong about the formatting is worth recovering. It does
    not tolerate guessing at the content.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("```")[1] if "```" in candidate[3:] else candidate[3:]
        candidate = candidate.removeprefix("json").strip()

    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return None, f"not JSON ({exc.msg})"
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as inner:
            return None, f"not JSON ({inner.msg})"

    if not isinstance(value, dict):
        return None, f"expected a JSON object, got {type(value).__name__}"
    return value, ""


def _backoff(attempt: int) -> float:
    """Exponential with jitter, so a fleet of workers does not retry in lockstep."""
    return min(2.0**attempt, 8.0) * (0.5 + random.random() / 2)


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)


def _add_costs(first: float | None, second: float | None) -> float | None:
    """None means unknown, and unknown plus anything is still unknown.

    Treating it as zero would make an unpriced repair look free.
    """
    if first is None or second is None:
        return None
    return first + second
