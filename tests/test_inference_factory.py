import pytest
from pydantic import ValidationError

from app.config import Settings
from app.inference.factory import create_llm_provider
from app.inference.ollama_provider import OllamaProvider


def test_factory_creates_ollama_provider() -> None:
    settings = Settings(
        _env_file=None,
        llm_provider="ollama",
        ollama_model="test-model",
    )

    provider = create_llm_provider(settings)

    assert isinstance(provider, OllamaProvider)
    assert provider.settings is settings
    assert provider.settings.ollama_model == "test-model"


def test_settings_rejects_unsupported_provider() -> None:
    with pytest.raises(ValidationError, match="llm_provider"):
        Settings(
            _env_file=None,
            llm_provider="unsupported",
        )


def test_provider_selection_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    settings = Settings(_env_file=None)

    provider = create_llm_provider(settings)

    assert settings.llm_provider == "ollama"
    assert isinstance(provider, OllamaProvider)
