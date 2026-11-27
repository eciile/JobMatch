import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from app.config import Settings
from app.inference.base import LLMProviderError
from app.inference.ollama_provider import OllamaProvider


def test_ollama_provider_generates_structured_text() -> None:
    client = AsyncMock()

    raw_content = '{"skills": ["Python", "Docker"]}'

    client.chat.return_value = SimpleNamespace(
        message=SimpleNamespace(content=raw_content)
    )

    settings = Settings(
        ollama_model="test-model",
        ollama_temperature=0.0,
        ollama_num_ctx=8192,
        ollama_num_predict=2048,
    )

    provider = OllamaProvider(
        settings=settings,
        client=client,
    )

    schema = {
        "type": "object",
        "properties": {
            "skills": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
    }

    result = asyncio.run(
        provider.generate(
            system_prompt="Extract technical skills.",
            user_prompt="Python and Docker are required.",
            schema=schema,
        )
    )

    assert result == raw_content

    client.chat.assert_awaited_once_with(
        model="test-model",
        messages=[
            {
                "role": "system",
                "content": "Extract technical skills.",
            },
            {
                "role": "user",
                "content": "Python and Docker are required.",
            },
        ],
        format=schema,
        options={
            "temperature": 0.0,
            "num_ctx": 8192,
            "num_predict": 2048,
        },
        think=False,
        stream=False,
        keep_alive="10m",
    )


def test_ollama_provider_handles_timeout() -> None:
    client = AsyncMock()
    client.chat.side_effect = httpx.ReadTimeout("Request timed out")

    provider = OllamaProvider(client=client)

    with pytest.raises(LLMProviderError, match="timeout"):
        asyncio.run(
            provider.generate(
                system_prompt="System prompt",
                user_prompt="User prompt",
                schema={"type": "object"},
            )
        )


def test_ollama_provider_handles_connection_failure() -> None:
    client = AsyncMock()
    client.chat.side_effect = httpx.ConnectError("Connection refused")

    provider = OllamaProvider(client=client)

    with pytest.raises(LLMProviderError, match="connect"):
        asyncio.run(
            provider.generate(
                system_prompt="System prompt",
                user_prompt="User prompt",
                schema={"type": "object"},
            )
        )


def test_ollama_provider_rejects_empty_response() -> None:
    client = AsyncMock()

    client.chat.return_value = SimpleNamespace(message=SimpleNamespace(content=""))

    provider = OllamaProvider(client=client)

    with pytest.raises(LLMProviderError, match="empty response"):
        asyncio.run(
            provider.generate(
                system_prompt="System prompt",
                user_prompt="User prompt",
                schema={"type": "object"},
            )
        )
