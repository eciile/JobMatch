from __future__ import annotations

import asyncio
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

from app.config import get_settings
from app.schemas import JobRequirements
from app.services.llm_job_extractor import (
    LlmJobExtractionError,
    OllamaJobExtractionClient,
)

DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "job_requirements.jsonl"

CATEGORIES = (
    "required_skills",
    "preferred_skills",
    "qualifications",
    "soft_skills",
    "languages",
)


@dataclass
class Metrics:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives

        if denominator == 0:
            return 0.0

        return self.true_positives / denominator

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives

        if denominator == 0:
            return 0.0

        return self.true_positives / denominator

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall

        if denominator == 0:
            return 0.0

        return 2 * self.precision * self.recall / denominator

    def accumulate(self, other: Metrics) -> None:
        self.true_positives += other.true_positives
        self.false_positives += other.false_positives
        self.false_negatives += other.false_negatives


def normalize(value: str) -> str:
    """Normalize superficial text differences."""

    value = unicodedata.normalize("NFKC", value)

    return " ".join(value.casefold().split())


def compare_requirements(
    expected: list[str],
    predicted: list[str],
) -> tuple[Metrics, set[str], set[str]]:
    """Compare requirements using normalized exact matching."""

    expected_set = {normalize(item) for item in expected}
    predicted_set = {normalize(item) for item in predicted}

    true_positives = expected_set & predicted_set
    false_positives = predicted_set - expected_set
    false_negatives = expected_set - predicted_set

    metrics = Metrics(
        true_positives=len(true_positives),
        false_positives=len(false_positives),
        false_negatives=len(false_negatives),
    )

    return metrics, false_positives, false_negatives


def print_metrics(metrics: Metrics) -> None:
    print(f"TP: {metrics.true_positives}")
    print(f"FP: {metrics.false_positives}")
    print(f"FN: {metrics.false_negatives}")
    print(f"Precision: {metrics.precision:.2%}")
    print(f"Recall: {metrics.recall:.2%}")
    print(f"F1-score: {metrics.f1:.2%}")


def load_dataset() -> list[dict]:
    rows = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if not rows:
        raise ValueError("Evaluation dataset is empty.")

    return rows


async def main() -> None:
    settings = get_settings()
    client = OllamaJobExtractionClient(settings=settings)

    dataset = load_dataset()

    global_metrics = Metrics()

    category_metrics = {category: Metrics() for category in CATEGORIES}

    successful_extractions = 0
    failed_extractions = 0
    exact_matches = 0

    latencies: list[float] = []
    successful_latencies: list[float] = []

    print("=== JOBMATCH EXTRACTION EVALUATION ===")
    print("Model:", settings.ollama_model)
    print("Host:", settings.ollama_host)
    print("Dataset size:", len(dataset))
    print("Matching method: normalized exact matching")

    for sample in dataset:
        print("\n" + "=" * 60)
        print(f"{sample['id']} — {sample['title']}")

        expected = JobRequirements.model_validate(sample["expected"])

        start = perf_counter()
        succeeded = False

        try:
            predicted = await client.extract_requirements(
                title=sample["title"],
                description=sample["description"],
            )

            successful_extractions += 1
            succeeded = True

        except LlmJobExtractionError as exc:
            failed_extractions += 1

            print("Extraction failed:", exc)

            # Failed extraction receives no predicted criteria.
            predicted = JobRequirements()

        elapsed = perf_counter() - start
        latencies.append(elapsed)

        if succeeded:
            successful_latencies.append(elapsed)

        print(f"Latency: {elapsed:.2f} s")

        print("\nPredicted requirements:")
        print(predicted.model_dump_json(indent=2))

        sample_metrics = Metrics()

        for category in CATEGORIES:
            metrics, unexpected, missing = compare_requirements(
                expected=getattr(expected, category),
                predicted=getattr(predicted, category),
            )

            category_metrics[category].accumulate(metrics)
            sample_metrics.accumulate(metrics)

            if unexpected or missing:
                print(f"\nCategory: {category}")

                if unexpected:
                    print("  Unexpected:", sorted(unexpected))

                if missing:
                    print("  Missing:", sorted(missing))

        global_metrics.accumulate(sample_metrics)

        is_exact_match = (
            succeeded
            and sample_metrics.false_positives == 0
            and sample_metrics.false_negatives == 0
        )

        if is_exact_match:
            exact_matches += 1

        print("\nSample metrics:")
        print_metrics(sample_metrics)
        print("Exact match:", is_exact_match)

    total = len(dataset)

    print("\n" + "=" * 60)
    print("=== GLOBAL RESULTS ===")

    print(f"Samples: {total}")
    print(f"Successful extractions: {successful_extractions}")
    print(f"Failed extractions: {failed_extractions}")
    print(f"Success rate: {successful_extractions / total:.2%}")
    print(f"Exact match rate: {exact_matches / total:.2%}")

    print(f"Mean latency (all): {mean(latencies):.2f} s")

    if successful_latencies:
        print(f"Mean latency (successful): {mean(successful_latencies):.2f} s")

    print("\nMicro-averaged metrics:")
    print_metrics(global_metrics)

    print("\n=== RESULTS BY CATEGORY ===")

    for category, metrics in category_metrics.items():
        print(f"\n{category}:")
        print_metrics(metrics)


if __name__ == "__main__":
    asyncio.run(main())
