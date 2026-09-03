"""OpenAI-compatible `/chat/completions`.

One client covers OpenAI, vLLM, Together, Groq, and a LiteLLM proxy, which is
why EvalLoop owns this rather than depending on a vendor SDK: full control over
the cache key, the retry policy, and the cost ledger, all three of which are
product surface rather than plumbing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx

from evalloop.contracts.judgeconf import JudgeConfig
from evalloop.contracts.protocols import RenderedPrompt
from evalloop.contracts.result import TokenUsage
from evalloop.judge.client import (
    ProviderError,
    ProviderResult,
    RateLimited,
    ServerError,
    Timeout,
)

__all__ = ["OpenAICompatProvider"]

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


@dataclass
class OpenAICompatProvider:
    name: str = "openai_compat"

    def complete(
        self,
        prompt: RenderedPrompt,
        schema: dict[str, Any],
        config: JudgeConfig,
    ) -> ProviderResult:
        base_url = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        api_key = _api_key(config)

        messages: list[dict[str, str]] = []
        if prompt.system:
            messages.append({"role": "system", "content": prompt.system})
        messages.extend(prompt.messages)

        body: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            # Schema-constrained decoding where the endpoint supports it. Where
            # it does not, the request is still accepted and the response is
            # parsed and repaired by the client instead.
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "evalloop_answer", "schema": schema, "strict": False},
            },
        }
        if config.top_p is not None:
            body["top_p"] = config.top_p

        try:
            response = httpx.post(
                f"{base_url}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=config.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise Timeout(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc

        _raise_for_status(response)

        payload = response.json()
        try:
            text = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {payload!r}") from exc

        return ProviderResult(text=text, usage=_usage(payload))


def _api_key(config: JudgeConfig) -> str:
    variable = config.api_key_env or "OPENAI_API_KEY"
    key = os.environ.get(variable)
    if not key:
        raise ProviderError(
            f"no API key: environment variable {variable} is unset. "
            f"Set it, or point judges.yaml at a different api_key_env."
        )
    return key


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 429:
        raise RateLimited(f"429: {response.text[:200]}")
    if response.status_code >= 500:
        raise ServerError(f"{response.status_code}: {response.text[:200]}")
    if response.status_code >= 400:
        # 4xx other than 429 is a bad request, not bad luck. Retrying it would
        # spend money re-sending something the provider has already refused.
        raise ProviderError(f"{response.status_code}: {response.text[:200]}")


def _usage(payload: dict[str, Any]) -> TokenUsage:
    """Token counts if reported. Cost stays None - pricing arrives in P2, and
    a made-up zero would read as a free call."""
    usage = payload.get("usage") or {}
    return TokenUsage(
        tokens_in=int(usage.get("prompt_tokens", 0)),
        tokens_out=int(usage.get("completion_tokens", 0)),
        cost_usd=None,
    )
