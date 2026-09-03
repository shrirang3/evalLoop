"""The two HTTP providers, against a mocked transport.

Everything else in the suite uses MockProvider, which means these two - the only
code that will ever talk to a real API - would otherwise ship untested. What
matters is the request shape (schema-constrained output is the whole point) and
the error mapping (whether something is retried is decided here).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from evalloop.contracts import JudgeConfig, RenderedPrompt
from evalloop.judge import (
    AnthropicProvider,
    OpenAICompatProvider,
    ProviderError,
    RateLimited,
    ServerError,
    Timeout,
)

SCHEMA = {"type": "object", "properties": {"answer": {"type": "boolean"}}}
PROMPT = RenderedPrompt(
    system="You audit transcripts.",
    messages=[{"role": "user", "content": "Did the agent follow policy?"}],
)

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


@pytest.fixture(autouse=True)
def _keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


def _openai_response(content: str = '{"answer": true}', **usage: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 120),
                "completion_tokens": usage.get("completion_tokens", 8),
            },
        },
    )


def _anthropic_response(payload: dict[str, Any] | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "content": [
                {"type": "tool_use", "name": "record_answer", "input": payload or {"answer": True}}
            ],
            "usage": {"input_tokens": 130, "output_tokens": 9},
        },
    )


# --- openai_compat ---


@respx.mock
def test_openai_sends_a_json_schema_constraint() -> None:
    """Schema-constrained decoding where the endpoint supports it. Without it
    the parse-and-repair path becomes the norm rather than the fallback."""
    route = respx.post(OPENAI_URL).mock(return_value=_openai_response())
    OpenAICompatProvider().complete(
        PROMPT, SCHEMA, JudgeConfig(provider="openai_compat", model="gpt-x")
    )

    body = json.loads(route.calls[0].request.content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA


@respx.mock
def test_openai_puts_the_system_prompt_in_the_message_list() -> None:
    route = respx.post(OPENAI_URL).mock(return_value=_openai_response())
    OpenAICompatProvider().complete(
        PROMPT, SCHEMA, JudgeConfig(provider="openai_compat", model="gpt-x")
    )

    messages = json.loads(route.calls[0].request.content)["messages"]
    assert messages[0] == {"role": "system", "content": "You audit transcripts."}
    assert messages[1]["content"] == "Did the agent follow policy?"


@respx.mock
def test_openai_omits_top_p_unless_configured() -> None:
    """Sending a default the user did not ask for changes the judge's behaviour
    and its version hash for no reason."""
    route = respx.post(OPENAI_URL).mock(return_value=_openai_response())
    config = JudgeConfig(provider="openai_compat", model="gpt-x")
    OpenAICompatProvider().complete(PROMPT, SCHEMA, config)
    assert "top_p" not in json.loads(route.calls[0].request.content)

    route.reset()
    OpenAICompatProvider().complete(PROMPT, SCHEMA, config.model_copy(update={"top_p": 0.9}))
    assert json.loads(route.calls[0].request.content)["top_p"] == 0.9


@respx.mock
def test_openai_honours_a_custom_base_url() -> None:
    """One client covers vLLM, Together, Groq, and a LiteLLM proxy."""
    route = respx.post("https://proxy.internal/v1/chat/completions").mock(
        return_value=_openai_response()
    )
    OpenAICompatProvider().complete(
        PROMPT,
        SCHEMA,
        JudgeConfig(provider="openai_compat", model="m", base_url="https://proxy.internal/v1/"),
    )
    assert route.called


@respx.mock
def test_openai_reports_token_usage_but_not_a_made_up_cost() -> None:
    """Pricing arrives in P2. A zero here would read as a free call."""
    respx.post(OPENAI_URL).mock(return_value=_openai_response(prompt_tokens=500))
    result = OpenAICompatProvider().complete(
        PROMPT, SCHEMA, JudgeConfig(provider="openai_compat", model="m")
    )
    assert result.usage.tokens_in == 500
    assert result.usage.cost_usd is None


@respx.mock
def test_openai_sends_the_bearer_token() -> None:
    route = respx.post(OPENAI_URL).mock(return_value=_openai_response())
    OpenAICompatProvider().complete(
        PROMPT, SCHEMA, JudgeConfig(provider="openai_compat", model="m")
    )
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-test"


def test_openai_missing_key_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """So the fix is obvious rather than a 401 from a vendor."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAICompatProvider().complete(
            PROMPT, SCHEMA, JudgeConfig(provider="openai_compat", model="m")
        )


def test_a_custom_api_key_env_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="MY_PROXY_KEY"):
        OpenAICompatProvider().complete(
            PROMPT,
            SCHEMA,
            JudgeConfig(provider="openai_compat", model="m", api_key_env="MY_PROXY_KEY"),
        )


@respx.mock
@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, RateLimited),
        (500, ServerError),
        (503, ServerError),
        (400, ProviderError),
        (401, ProviderError),
        (404, ProviderError),
    ],
)
def test_openai_status_codes_map_to_the_right_retry_decision(
    status: int, expected: type[Exception]
) -> None:
    """The class decides whether the client retries. A 4xx other than 429 is a
    bad request, and retrying it spends money re-sending something already
    refused."""
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(status, text="nope"))
    with pytest.raises(expected):
        OpenAICompatProvider().complete(
            PROMPT, SCHEMA, JudgeConfig(provider="openai_compat", model="m")
        )


@respx.mock
def test_rate_limited_is_not_caught_by_the_generic_4xx_branch() -> None:
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(429, text="slow down"))
    with pytest.raises(RateLimited):
        OpenAICompatProvider().complete(
            PROMPT, SCHEMA, JudgeConfig(provider="openai_compat", model="m")
        )


@respx.mock
def test_openai_timeout_maps_to_timeout() -> None:
    respx.post(OPENAI_URL).mock(side_effect=httpx.ReadTimeout("too slow"))
    with pytest.raises(Timeout):
        OpenAICompatProvider().complete(
            PROMPT, SCHEMA, JudgeConfig(provider="openai_compat", model="m")
        )


@respx.mock
def test_openai_connection_failure_is_not_retried_as_a_server_error() -> None:
    respx.post(OPENAI_URL).mock(side_effect=httpx.ConnectError("no route"))
    with pytest.raises(ProviderError):
        OpenAICompatProvider().complete(
            PROMPT, SCHEMA, JudgeConfig(provider="openai_compat", model="m")
        )


@respx.mock
def test_openai_null_content_becomes_empty_text_not_a_crash() -> None:
    """A refusal or a max-tokens stop can return null content. That belongs in
    the invalid-output rate, not in a traceback."""
    respx.post(OPENAI_URL).mock(
        return_value=httpx.Response(200, json={"choices": [{"message": {"content": None}}]})
    )
    result = OpenAICompatProvider().complete(
        PROMPT, SCHEMA, JudgeConfig(provider="openai_compat", model="m")
    )
    assert result.text == ""


@respx.mock
def test_openai_unexpected_shape_is_a_provider_error() -> None:
    respx.post(OPENAI_URL).mock(return_value=httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(ProviderError, match="unexpected response shape"):
        OpenAICompatProvider().complete(
            PROMPT, SCHEMA, JudgeConfig(provider="openai_compat", model="m")
        )


# --- anthropic ---


@respx.mock
def test_anthropic_forces_the_schema_as_a_tool() -> None:
    """Anthropic's mechanism for guaranteed structure. tool_choice makes the
    model answer by calling it, so parse-and-repair stays a fallback."""
    route = respx.post(ANTHROPIC_URL).mock(return_value=_anthropic_response())
    AnthropicProvider().complete(
        PROMPT, SCHEMA, JudgeConfig(provider="anthropic", model="claude-sonnet-5")
    )

    body = json.loads(route.calls[0].request.content)
    assert body["tools"][0]["input_schema"] == SCHEMA
    assert body["tool_choice"] == {"type": "tool", "name": "record_answer"}


@respx.mock
def test_anthropic_sends_the_system_prompt_as_a_top_level_field() -> None:
    """Not as a message. Anthropic's API is shaped differently, which is the
    reason the provider interface exists."""
    route = respx.post(ANTHROPIC_URL).mock(return_value=_anthropic_response())
    AnthropicProvider().complete(
        PROMPT, SCHEMA, JudgeConfig(provider="anthropic", model="claude-sonnet-5")
    )

    body = json.loads(route.calls[0].request.content)
    assert body["system"] == "You audit transcripts."
    assert all(m["role"] != "system" for m in body["messages"])


@respx.mock
def test_anthropic_sends_the_api_key_and_version_headers() -> None:
    route = respx.post(ANTHROPIC_URL).mock(return_value=_anthropic_response())
    AnthropicProvider().complete(
        PROMPT, SCHEMA, JudgeConfig(provider="anthropic", model="claude-sonnet-5")
    )

    headers = route.calls[0].request.headers
    assert headers["x-api-key"] == "sk-ant-test"
    assert headers["anthropic-version"] == "2023-06-01"


@respx.mock
def test_anthropic_returns_the_tool_input_as_json_text() -> None:
    """Returned as a string so the client's parse-and-repair stays the single
    place that turns any vendor's reply into a parsed answer."""
    respx.post(ANTHROPIC_URL).mock(
        return_value=_anthropic_response({"answer": False, "reason": "outside window"})
    )
    result = AnthropicProvider().complete(
        PROMPT, SCHEMA, JudgeConfig(provider="anthropic", model="claude-sonnet-5")
    )
    assert json.loads(result.text) == {"answer": False, "reason": "outside window"}
    assert result.usage.tokens_in == 130
    assert result.usage.cost_usd is None


@respx.mock
def test_anthropic_falls_back_to_a_text_block() -> None:
    """Forced tool use should make this unreachable, but a refusal can still
    return prose. Handing it to the parser records invalid_output rather than
    crashing the run."""
    respx.post(ANTHROPIC_URL).mock(
        return_value=httpx.Response(
            200, json={"content": [{"type": "text", "text": "I cannot answer that."}]}
        )
    )
    result = AnthropicProvider().complete(
        PROMPT, SCHEMA, JudgeConfig(provider="anthropic", model="claude-sonnet-5")
    )
    assert result.text == "I cannot answer that."


@respx.mock
def test_anthropic_no_usable_block_is_a_provider_error() -> None:
    respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(200, json={"content": []}))
    with pytest.raises(ProviderError, match="no usable content block"):
        AnthropicProvider().complete(
            PROMPT, SCHEMA, JudgeConfig(provider="anthropic", model="claude-sonnet-5")
        )


@respx.mock
@pytest.mark.parametrize(
    ("status", "expected"),
    [(429, RateLimited), (529, ServerError), (400, ProviderError)],
)
def test_anthropic_status_codes_map_to_the_right_retry_decision(
    status: int, expected: type[Exception]
) -> None:
    respx.post(ANTHROPIC_URL).mock(return_value=httpx.Response(status, text="nope"))
    with pytest.raises(expected):
        AnthropicProvider().complete(
            PROMPT, SCHEMA, JudgeConfig(provider="anthropic", model="claude-sonnet-5")
        )


@respx.mock
def test_anthropic_timeout_maps_to_timeout() -> None:
    respx.post(ANTHROPIC_URL).mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(Timeout):
        AnthropicProvider().complete(
            PROMPT, SCHEMA, JudgeConfig(provider="anthropic", model="claude-sonnet-5")
        )


@respx.mock
def test_anthropic_connection_failure_is_a_provider_error() -> None:
    respx.post(ANTHROPIC_URL).mock(side_effect=httpx.ConnectError("no route"))
    with pytest.raises(ProviderError):
        AnthropicProvider().complete(
            PROMPT, SCHEMA, JudgeConfig(provider="anthropic", model="claude-sonnet-5")
        )


def test_anthropic_missing_key_names_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider().complete(
            PROMPT, SCHEMA, JudgeConfig(provider="anthropic", model="claude-sonnet-5")
        )


@respx.mock
def test_anthropic_honours_a_custom_base_url() -> None:
    route = respx.post("https://gateway.internal/v1/messages").mock(
        return_value=_anthropic_response()
    )
    AnthropicProvider().complete(
        PROMPT,
        SCHEMA,
        JudgeConfig(provider="anthropic", model="m", base_url="https://gateway.internal/v1"),
    )
    assert route.called


@respx.mock
def test_anthropic_sends_top_p_only_when_configured() -> None:
    route = respx.post(ANTHROPIC_URL).mock(return_value=_anthropic_response())
    config = JudgeConfig(provider="anthropic", model="m")
    AnthropicProvider().complete(PROMPT, SCHEMA, config)
    assert "top_p" not in json.loads(route.calls[0].request.content)

    route.reset()
    AnthropicProvider().complete(PROMPT, SCHEMA, config.model_copy(update={"top_p": 0.8}))
    assert json.loads(route.calls[0].request.content)["top_p"] == 0.8


@respx.mock
def test_a_prompt_with_no_system_omits_the_field() -> None:
    route = respx.post(ANTHROPIC_URL).mock(return_value=_anthropic_response())
    AnthropicProvider().complete(
        RenderedPrompt(system=None, messages=[{"role": "user", "content": "ok?"}]),
        SCHEMA,
        JudgeConfig(provider="anthropic", model="m"),
    )
    assert "system" not in json.loads(route.calls[0].request.content)
