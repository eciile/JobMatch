from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings, get_settings
from app.inference.base import LLMProvider, LLMProviderError
from app.inference.factory import create_llm_provider
from app.schemas import (
    ExtractedJobPosting,
    GenericJobContent,
    JobRequirements,
)
from app.text_utils import repair_utf8_mojibake

MAX_INPUT_CHARACTERS = 8_000
REQUIREMENTS_SECTION_START_MARKERS = (
    "minimum qualifications",
    "required qualifications",
    "basic qualifications",
    "what we're looking for",
    "what we are looking for",
    "what you bring",
    "what you'll bring",
    "what you will bring",
    "it would be great if you have",
    "skills and experience",
    "your qualifications",
    "your profile",
    "about you",
    "who you are",
    "requirements",
    "\nprofil\n",
    "\nprofil recherché",
    "\nprofil souhaité",
    "\nvotre profil",
    "ce que vous apportez",
    "compétences requises",
    "qualifications requises",
)
REQUIREMENTS_SECTION_END_MARKERS = (
    "what's in it for you",
    "what’s in it for you",
    "what we offer",
    "our benefits",
    "benefits and perks",
    "next steps",
    "interview process",
    "about the company",
    "about us",
    "equal opportunity",
    "critères de l'offre",
    "critères de l’offre",
    "l'employeur",
    "l’employeur",
    "ce que nous offrons",
    "nos avantages",
)
OPTIONAL_REQUIREMENTS_MARKERS = (
    "nice to have",
    "good to have",
    "preferred qualifications",
    "preferred skills",
    "bonus points",
    "as a bonus",
    "compétences appréciées",
    "compétences souhaitées",
    "atouts supplémentaires",
)
SYSTEM_PROMPT = """
You extract structured information from job advertisements.

Rules:
- Use only information explicitly present in the provided content.
- Never invent company names, locations, or employment types.
- Application URLs and dates are managed by the application.
- Do not generate application URLs, publication dates,
  or expiration dates.
- Extract employment type whenever terms such as full-time,
  part-time, permanent, temporary, contract, internship,
  apprenticeship, freelance, or fixed-term are explicitly stated.
- Put mandatory technical skills, tools, frameworks, platforms,
  and methodologies under required_skills.
- Put optional, desirable, preferred, or bonus technical skills
  under preferred_skills.
- Put degrees, certifications, licences, years of experience,
  and other formal eligibility criteria under qualifications.
- Put explicitly requested interpersonal or behavioural abilities
  such as communication, teamwork, autonomy, and leadership under
  soft_skills.
- Put explicitly requested spoken or written languages under
  languages.
- Do not classify programming languages such as Python,
  Java, or SQL as spoken languages.
- Do not infer requirements that are not explicitly stated.
- Include soft skills only when the advertisement presents them
  as an explicit requirement.
- Candidate evidence rule: include an item only when the advertisement
  directly asks the candidate to have it. Typical evidence appears under
  headings such as requirements, qualifications, what you bring, you have,
  your profile, or nice to have.
- Technologies mentioned only in company or team descriptions, existing
  stack lists, product architecture, responsibilities, examples, benefits,
  interview steps, or employer boilerplate are context, not requirements.
- A responsibility involving a technology does not by itself prove that
  prior skill in that technology is required.
- Never turn a responsibility into an inferred soft skill. For example,
  "collaborate with other teams" does not imply "collaboration" belongs
  under soft_skills unless collaboration is explicitly requested as a trait.
- Required skills include wording such as:
  required, must have, essential, indispensable,
  maîtrise requise, obligatoire, or prerequisites.
- Preferred skills include wording such as:
  preferred, desirable, bonus, nice to have,
  apprécié, souhaité, idéalement, or un plus.
- Preserve common technology names such as Python,
  PostgreSQL, FastAPI, Docker, AWS, and Apache Airflow.
- Do not place the same skill in both required_skills
  and preferred_skills.
- Preserve the wording used in the source text.
- required_skills and preferred_skills must contain only
  short, atomic skill or technology names.
- Never return full sentences, qualification statements,
  responsibilities, degrees, or experience descriptions
  inside skill lists.
- Split combined requirements into separate items.
- For example, "Java or Python development using APIs and
  microservices" becomes ["Java", "Python", "REST API",
  "Microservices"]. and "Maîtrise de Java ou Python, des API et des microservices"
  becomes ["Java", "Python", "REST API", "Microservices"].
- Keep each skill concise, normally between one and five words.
- Degrees, certifications, and years of experience belong under
  qualifications, never under skill lists.
- Do not place the same item in multiple requirement categories.
- Do not put the same skill in both required_skills and
  preferred_skills.
- Return null for unavailable optional values.
- Return an empty list when employment type is unavailable.
- The original description is preserved by the application.
- Do not generate a description.
- Return only data matching the requested JSON schema.
""".strip()


class LlmJobExtractionError(ValueError):
    """raised when the LLM fails to produce a valid job posting"""


class LlmExtractedJob(BaseModel):
    """Internal generation schema for LLM-based job extraction."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    company: str | None = None
    location: str | None = None
    employment_types: list[str] = Field(default_factory=list)
    requirements: JobRequirements = Field(
        default_factory=JobRequirements,
    )


class CategorizedRequirement(BaseModel):
    """One atomic requirement classified by its public output field."""

    value: str = Field(min_length=1)
    category: Literal[
        "required_skills",
        "preferred_skills",
        "qualifications",
        "soft_skills",
        "languages",
    ]


class CategorizedRequirements(BaseModel):
    """Internal LLM response that avoids parallel defaulted arrays."""

    items: list[CategorizedRequirement]


class JobExtractionService:
    """Extract structured job data using an inference provider."""

    def __init__(
        self,
        provider: LLMProvider,
        settings: Settings | None = None,
    ) -> None:
        self.provider = provider
        self.settings = settings or get_settings()

    async def extract_job(
        self,
        content: GenericJobContent,
    ) -> ExtractedJobPosting:
        """Convert cleaned page content into structured job data."""

        schema = LlmExtractedJob.model_json_schema()
        prompt = _build_prompt(
            content,
            max_input_characters=self.settings.llm_max_input_characters,
        )

        try:
            raw_content = await self.provider.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                schema=schema,
            )
        except LLMProviderError as exc:
            raise LlmJobExtractionError(str(exc)) from exc

        raw_content = repair_utf8_mojibake(raw_content)

        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise LlmJobExtractionError(
                "The inference provider returned invalid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise LlmJobExtractionError(
                "The inference provider returned an unexpected JSON structure."
            )

        # The source URL is controlled by JobMatch. Do not trust the
        # model to generate or reproduce an application URL.
        try:
            generated_job = LlmExtractedJob.model_validate(payload)

            job = ExtractedJobPosting.model_validate(
                {
                    **generated_job.model_dump(),
                    "description": content.text,
                    "application_url": content.source_url,
                    "date_posted": content.metadata.get("date_posted"),
                    "valid_through": content.metadata.get("valid_through"),
                }
            )

        except ValidationError as exc:
            raise LlmJobExtractionError(
                "The inference provider returned job data that failed validation."
            ) from exc

        generic_title_markers = {
            "détails offre",
            "detail offre",
            "job details",
            "offre d'emploi",
            "careers",
            "carrières",
            "recrutement",
        }

        normalized_title = job.title.casefold()

        if any(marker in normalized_title for marker in generic_title_markers):
            raise LlmJobExtractionError(
                "The inference provider did not extract a specific job title."
            )

        return job

    async def extract_requirements(
        self,
        *,
        title: str,
        description: str,
    ) -> JobRequirements:
        """Extract atomic requirements from an existing job."""

        requirements_text = _select_requirements_text(description)
        requirements_text = _label_requirement_priority(requirements_text)

        prompt = f"""
        Return every explicit candidate criterion as an atomic categorized
        item.

        Job title:
        {title}

        Candidate criteria:
        --- BEGIN CRITERIA ---
        {requirements_text[: self.settings.llm_max_input_characters]}     
        --- END CRITERIA ---

        Categories:
        - required_skills: named technologies and practical technical
          abilities in MAIN CANDIDATE CRITERIA.
        - preferred_skills: named technologies and practical technical
          abilities in OPTIONAL / NICE-TO-HAVE CRITERIA.
        - qualifications: experience, durations, degrees, certifications,
          licences, academic or professional background, and theoretical
          knowledge. These are qualifications even when technically themed.
        - soft_skills: explicit behavioural or interpersonal traits.
        - languages: explicit human-language requirements only.

        Classification examples:
        - "multiple years of Python experience" yields two items: Python as
          required_skills and the full experience criterion as qualifications.
        - "strong computer science background" is qualifications.
        - "LLM and AI Engineering theoretical knowledge" yields three items:
          LLM Engineering and AI Engineering as required_skills, plus the
          knowledge criterion as qualifications.
        - "good communication skills" is soft_skills.
        - "PyTorch experience" in the OPTIONAL section yields PyTorch as
          preferred_skills.

        For every qualification with a named technical subject, also emit
        that subject as an atomic technical skill. Use required_skills for a
        MAIN criterion and preferred_skills for an OPTIONAL criterion.
        Keep the qualification item even when its technical subject was also
        emitted as a skill.

        Expand shared suffixes into complete atomic names. For example,
        "LLM / AI Engineering" must produce "LLM Engineering" and
        "AI Engineering", never the incomplete value "LLM".

        Normalize qualification wording without changing its meaning:
        - "first experience in X" becomes "X experience".
        - "at least N years of experience programming in X" becomes
          "N+ years X experience".
        - Remove evaluative modifiers such as "good" or "strong" from
          background and theoretical-knowledge qualifications.

        Account for every explicit criterion. Preserve exact numbers and do
        not copy durations from examples. Split distinct technical skills
        into atomic items. Preserve alternatives within formal
        qualifications. Do not output both a short and long version of the
        same criterion within one category. Do not infer missing criteria.
        """.strip()

        try:
            raw_content = await self.provider.generate(
                system_prompt=(
                    "Classify explicit candidate criteria into "
                    "the requested JSON fields without omissions."
                ),
                user_prompt=prompt,
                schema=CategorizedRequirements.model_json_schema(),
            )
        except LLMProviderError as exc:
            raise LlmJobExtractionError(str(exc)) from exc

        raw_content = repair_utf8_mojibake(raw_content)

        try:
            payload = json.loads(raw_content)
            categorized = CategorizedRequirements.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise LlmJobExtractionError(
                "The inference provider returned invalid job requirements."
            ) from exc

        grouped: dict[str, list[str]] = {
            "required_skills": [],
            "preferred_skills": [],
            "qualifications": [],
            "soft_skills": [],
            "languages": [],
        }

        for item in categorized.items:
            grouped[item.category].append(item.value)

        return JobRequirements.model_validate(grouped)


def _select_requirements_text(description: str) -> str:
    """
    Prefer an explicit candidate-requirements section when present.

    ATS descriptions are often flattened into one long string. This keeps
    company stack and benefits sections from being presented to the model as
    candidate requirements while retaining the full text as a fallback.
    """

    normalized = description.casefold()
    start_positions = [
        position
        for marker in REQUIREMENTS_SECTION_START_MARKERS
        if (position := normalized.find(marker)) >= 0
    ]

    if not start_positions:
        return description

    start = min(start_positions)
    end_positions = [
        position
        for marker in REQUIREMENTS_SECTION_END_MARKERS
        if (position := normalized.find(marker, start + 1)) >= 0
    ]
    end = min(end_positions) if end_positions else len(description)

    return description[start:end].strip()


def _label_requirement_priority(text: str) -> str:
    """Label an optional subsection so small models preserve priority."""

    normalized = text.casefold()
    optional_positions = [
        (position, marker)
        for marker in OPTIONAL_REQUIREMENTS_MARKERS
        if (position := normalized.find(marker)) > 0
    ]

    if not optional_positions:
        return text

    position, marker = min(optional_positions)
    main_requirements = text[:position].strip()
    optional_requirements = text[position + len(marker) :].strip(" :.-")

    return (
        "MAIN CANDIDATE CRITERIA:\n"
        f"{main_requirements}\n\n"
        "OPTIONAL / NICE-TO-HAVE CRITERIA:\n"
        f"{optional_requirements}"
    )


def _build_prompt(
    content: GenericJobContent,
    max_input_characters: int = MAX_INPUT_CHARACTERS,
) -> str:
    page_text = content.text[:max_input_characters]
    page_title = content.page_title or "Unavailable"

    metadata_lines = [f"- {key}: {value}" for key, value in content.metadata.items()]

    metadata_text = (
        "\n".join(metadata_lines)
        if metadata_lines
        else "- No structured metadata found"
    )

    return f"""
    Extract the job advertisement into structured data.

    Page title:
    {page_title}

    Page metadata:
    {metadata_text}

    Main job content:
    --- BEGIN JOB CONTENT ---
    {page_text}
    --- END JOB CONTENT ---

    Rules:
    - Use only explicitly provided information.
    - The title must not be empty.
    - Empty strings are forbidden.
    - Use null for missing optional string fields.
    - Preserve company and location metadata when supplied.
    - Map CDI to PERMANENT.
    - Map CDD to FIXED_TERM.
    - Map stage to INTERNSHIP.
    - Map alternance to APPRENTICESHIP.
    """.strip()


@lru_cache
def get_llm_job_extraction_client() -> JobExtractionService:
    """Return one reusable job extraction service."""

    settings = get_settings()

    return JobExtractionService(
        provider=create_llm_provider(settings),
        settings=settings,
    )
