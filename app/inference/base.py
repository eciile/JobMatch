from typing import Any, Protocol


class LLMProviderError(RuntimeError):
    """Raised when an inference provider fails."""


class LLMProvider(Protocol):
    """Common contract for structured LLM generation."""

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
    ) -> str:
        """Generate non-empty text constrained by the supplied JSON schema.

        Raises:
            LLMProviderError: If generation fails or returns empty text.
        """
        ...
