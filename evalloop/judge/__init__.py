"""Judge client and providers."""

from evalloop.judge.cache import PostgresCache
from evalloop.judge.client import (
    JudgeClient,
    Provider,
    ProviderError,
    ProviderResult,
    RateLimited,
    ServerError,
    Timeout,
    cache_key,
)
from evalloop.judge.providers.anthropic import AnthropicProvider
from evalloop.judge.providers.mock import MockProvider
from evalloop.judge.providers.openai_compat import OpenAICompatProvider

__all__ = [
    "AnthropicProvider",
    "JudgeClient",
    "MockProvider",
    "OpenAICompatProvider",
    "PostgresCache",
    "Provider",
    "ProviderError",
    "ProviderResult",
    "RateLimited",
    "ServerError",
    "Timeout",
    "cache_key",
    "make_provider",
]

_PROVIDERS: dict[str, type] = {
    "anthropic": AnthropicProvider,
    "openai_compat": OpenAICompatProvider,
    "mock": MockProvider,
}


def make_provider(name: str) -> Provider:
    """Build a provider by the name used in judges.yaml."""
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        raise ValueError(
            f"unknown judge provider {name!r}; available: {', '.join(sorted(_PROVIDERS))}"
        ) from None
    provider: Provider = factory()
    return provider
