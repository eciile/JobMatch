from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """configuration settings loaded from environment variables"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    ollama_timeout_seconds: float = 60.0
    ollama_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
    )

    ollama_num_ctx: int = Field(
        default=8192,
        gt=0,
    )

    ollama_num_predict: int = Field(
        default=2048,
        gt=0,
    )

    llm_max_input_characters: int = Field(
        default=8000,
        ge=200,
    )


@lru_cache
def get_settings() -> Settings:
    """returns a cached instance of the Settings class"""
    return Settings()
