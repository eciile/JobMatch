from app.config import Settings, get_settings
from app.inference.base import LLMProvider
from app.inference.ollama_provider import OllamaProvider


def create_llm_provider(
    settings: Settings | None = None,
) -> LLMProvider:
    """Create the configured inference provider."""

    settings = settings or get_settings()

    if settings.llm_provider == "ollama":
        return OllamaProvider(settings=settings)

    raise ValueError(f"Unsupported inference provider: {settings.llm_provider}")
