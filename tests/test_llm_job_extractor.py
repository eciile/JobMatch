import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.services.llm_job_extractor import (
    LlmExtractedJob,
    OllamaJobExtractionClient,
    _label_requirement_priority,
    _select_requirements_text,
)


def test_extract_requirements_returns_validated_requirements() -> None:
    class FakeOllamaClient:
        def __init__(self) -> None:
            self.request: dict | None = None

        async def chat(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "items": [
                                {
                                    "value": "Python",
                                    "category": "required_skills",
                                },
                                {
                                    "value": "Python",
                                    "category": "required_skills",
                                },
                                {
                                    "value": "Docker",
                                    "category": "preferred_skills",
                                },
                                {
                                    "value": "Python",
                                    "category": "preferred_skills",
                                },
                                {
                                    "value": "3 years experience",
                                    "category": "qualifications",
                                },
                                {
                                    "value": "Communication",
                                    "category": "soft_skills",
                                },
                                {
                                    "value": "English",
                                    "category": "languages",
                                },
                            ]
                        }
                    )
                )
            )

    ollama_client = FakeOllamaClient()
    extractor = OllamaJobExtractionClient(
        settings=Settings(ollama_model="test-model"),
        client=ollama_client,
    )

    requirements = asyncio.run(
        extractor.extract_requirements(
            title="Data Engineer",
            description="Python is required. Docker is a plus.",
        )
    )

    assert requirements.required_skills == ["Python"]
    assert requirements.preferred_skills == ["Docker"]
    assert requirements.qualifications == ["3 years experience"]
    assert requirements.soft_skills == ["Communication"]
    assert requirements.languages == ["English"]
    assert ollama_client.request is not None
    assert ollama_client.request["model"] == "test-model"
    assert "Data Engineer" in (ollama_client.request["messages"][1]["content"])
    prompt = ollama_client.request["messages"][1]["content"]
    assert "Return every explicit candidate criterion" in prompt
    assert "MAIN CANDIDATE CRITERIA" in prompt
    assert "OPTIONAL / NICE-TO-HAVE CRITERIA" in prompt
    assert "Account for every explicit criterion" in prompt
    assert "named technical subject" in prompt
    assert "N+ years X experience" in prompt
    assert 'never the incomplete value "LLM"' in prompt
    assert ollama_client.request["format"]["title"] == ("CategorizedRequirements")
    assert "items" in ollama_client.request["format"]["properties"]


def test_select_requirements_text_excludes_company_stack() -> None:
    description = (
        "About the team: Our stack uses Java, Kafka, and Docker. "
        "You will build detection products. "
        "It would be great if you have: 2 years programming in "
        "Python. Strong computer science background. "
        "Nice to have: Vertex AI, AWS Bedrock, and PyTorch. "
        "What's in it for you? Health benefits and team events."
    )

    selected = _select_requirements_text(description)

    assert selected.startswith("It would be great if you have")
    assert "Python" in selected
    assert "Vertex AI" in selected
    assert "Java" not in selected
    assert "Health benefits" not in selected

    labeled = _label_requirement_priority(selected)

    assert "MAIN CANDIDATE CRITERIA:" in labeled
    assert "OPTIONAL / NICE-TO-HAVE CRITERIA:" in labeled
    assert "Vertex AI, AWS Bedrock, and PyTorch" in labeled


def test_llm_default_generation_options() -> None:
    settings = Settings(
        _env_file=None,
        ollama_temperature=0.0,
        ollama_num_ctx=8192,
        ollama_num_predict=2048,
        llm_max_input_characters=8000,
    )

    client = OllamaJobExtractionClient(settings=settings)

    assert client._generation_options() == {
        "temperature": 0.0,
        "num_ctx": 8192,
        "num_predict": 2048,
    }


def test_llm_custom_generation_options() -> None:
    settings = Settings(
        _env_file=None,
        ollama_temperature=0.2,
        ollama_num_ctx=8192,
        ollama_num_predict=2048,
        llm_max_input_characters=8000,
    )

    client = OllamaJobExtractionClient(settings=settings)

    assert client._generation_options() == {
        "temperature": 0.2,
        "num_ctx": 8192,
        "num_predict": 2048,
    }

    assert settings.llm_max_input_characters == 8000


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ollama_temperature", -0.1),
        ("ollama_temperature", 2.1),
        ("ollama_num_ctx", 0),
        ("ollama_num_predict", 0),
        ("llm_max_input_characters", 199),
    ],
)
def test_llm_rejects_invalid_configuration(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{field: value})


def test_llm_generation_schema_excludes_deterministic_fields() -> None:
    schema = LlmExtractedJob.model_json_schema()
    properties = schema["properties"]

    assert "application_url" not in properties
    assert "date_posted" not in properties
    assert "valid_through" not in properties
    assert "description" not in properties

    assert schema["additionalProperties"] is False


def test_llm_generation_schema_retains_semantic_fields() -> None:
    properties = LlmExtractedJob.model_json_schema()["properties"]

    assert set(properties) == {
        "title",
        "company",
        "location",
        "employment_types",
        "requirements",
    }


def test_llm_generation_schema_rejects_unexpected_fields() -> None:
    with pytest.raises(ValidationError):
        LlmExtractedJob.model_validate(
            {
                "title": "AI Engineer",
                "application_url": "https://example.com/job",
                "valid_through": "2026-12-31",
            }
        )
