"""Anthropic Messages API, behind the same interface.

Schema conformance uses forced tool use rather than a response-format flag: the
schema is declared as a single tool and `tool_choice` requires it, so the model
answers by calling it. That is Anthropic's mechanism for guaranteed structure,
and it means the parse-and-repair path is a fallback here rather than the norm.
"""

from __future__ import annotations

import json
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

__all__ = ["AnthropicProvider"]

_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
_API_VERSION = "2023-06-01"
_TOOL_NAME = "record_answer"


@dataclass
class AnthropicProvider:
    name: str = "anthropic"

    def complete(
        self,
        prompt: RenderedPrompt,
        schema: dict[str, Any],
        config: JudgeConfig,
    ) -> ProviderResult:
        base_url = (config.base_url or _DEFAULT_BASE_URL).rstrip("/")
        api_key = _api_key(config)

        body: dict[str, Any] = {
            "model": config.model,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "messages": prompt.messages,
            "tools": [
                {
                    "name": _TOOL_NAME,
                    "description": "Record your answer in the required structure.",
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": _TOOL_NAME},
        }
        if prompt.system:
            body["system"] = prompt.system
        if config.top_p is not None:
            body["top_p"] = config.top_p

        try:
            response = httpx.post(
                f"{base_url}/messages",
                json=body,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": _API_VERSION,
                    "content-type": "application/json",
                },
                timeout=config.timeout_s,
            )
        except httpx.TimeoutException as exc:
            raise Timeout(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(str(exc)) from exc

        _raise_for_status(response)
        payload = response.json()
        return ProviderResult(text=_extract(payload), usage=_usage(payload))


def _extract(payload: dict[str, Any]) -> str:
    """Pull the tool input out, or fall back to text.

    Returned as a JSON string so the client's parse-and-repair path stays the
    single place that turns a provider reply into a parsed answer, regardless of
    which vendor produced it.
    """
    blocks = payload.get("content") or []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            return json.dumps(block.get("input", {}))

    # Forced tool use should make this unreachable, but a refusal or a
    # max-tokens stop can still yield prose. Handing it to the parser produces a
    # recorded invalid_output rather than a crash.
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            return str(block.get("text", ""))
    raise ProviderError(f"no usable content block: {payload!r}")


def _api_key(config: JudgeConfig) -> str:
    variable = config.api_key_env or "ANTHROPIC_API_KEY"
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
        raise ProviderError(f"{response.status_code}: {response.text[:200]}")


def _usage(payload: dict[str, Any]) -> TokenUsage:
    usage = payload.get("usage") or {}
    return TokenUsage(
        tokens_in=int(usage.get("input_tokens", 0)),
        tokens_out=int(usage.get("output_tokens", 0)),
        cost_usd=None,
    )
