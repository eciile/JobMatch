from typing import Any

import httpx
from ollama import AsyncClient, ResponseError

from app.config import Settings, get_settings
from app.inference.base import LLMProviderError


class OllamaProvider:
    """Adapter for structured generation using an Ollama server."""

    def __init__(
        self,
        settings: Settings | None = None,
        client: AsyncClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        self.client = (
            client
            if client is not None
            else AsyncClient(
                host=self.settings.ollama_host,
                timeout=self.settings.ollama_timeout_seconds,
            )
        )

    def _generation_options(self) -> dict[str, int | float]:
        return {
            "temperature": self.settings.ollama_temperature,
            "num_ctx": self.settings.ollama_num_ctx,
            "num_predict": self.settings.ollama_num_predict,
        }

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> str:
        """Generate structured text using the configured Ollama model."""

        try:
            response = await self.client.chat(
                model=self.settings.ollama_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                format=schema,
                options=self._generation_options(),
                think=False,
                stream=False,
                keep_alive="10m",
            )

        except ResponseError as exc:
            raise LLMProviderError(f"Ollama returned an error: {exc.error}") from exc

        except httpx.TimeoutException as exc:
            raise LLMProviderError(
                "Ollama did not finish generation before the timeout."
            ) from exc

        except (httpx.ConnectError, ConnectionError) as exc:
            raise LLMProviderError("Could not connect to the Ollama server.") from exc

        except httpx.HTTPError as exc:
            raise LLMProviderError(
                f"Ollama request failed: {type(exc).__name__}."
            ) from exc

        raw_content = response.message.content

        if not raw_content or not raw_content.strip():
            raise LLMProviderError("Ollama returned an empty response.")

        return raw_content
