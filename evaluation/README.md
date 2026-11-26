# JobMatch Extraction Evaluation

## Objective

Evaluate the accuracy, reliability and latency of LLM-based job requirement extraction independently of webpage retrieval and candidate matching.

## Dataset

The initial development dataset contains six synthetic French and English job advertisements with manually annotated expected requirements.

Each sample contains:

- A unique identifier.
- The advertisement language.
- The job title and description.
- Expected required skills, preferred skills, qualifications, soft skills and languages.

## Execution

Run from the repository root with Ollama running and the configured model available:

```bash
python -m evaluation.evaluate_extraction
```

## Metrics

- Micro-averaged precision, recall and F1-score.
- Per-category extraction metrics.
- Extraction success rate.
- Exact match rate.
- Mean extraction latency.
- False positives and false negatives.

## Initial baseline

Model: `qwen3:4b-instruct-2507-q4_K_M`

Hardware: NVIDIA GeForce RTX 5060 8 GB.

| Metric | Baseline V0 |
|---|---:|
| Precision | 57.14% |
| Recall | 71.43% |
| F1-score | 63.49% |
| Extraction success rate | 100% |
| Exact match rate | 0% |
| Mean latency | 2.98 s |

Baseline results are stored in `results/baseline_v0.json`.

## Limitations

The initial dataset is small and synthetic. Results must not be interpreted as general real-world extraction accuracy.

Normalized exact matching may penalize semantically equivalent expressions. Extraction success indicates successful parsing and validation, not semantic correctness.

Latency measurements may include model loading and request overhead.

Live evaluation requires a running inference server and is kept separate from deterministic automated tests.

## Future improvements

- Expand French and English real-world evaluation samples.
- Evaluate metadata extraction independently.
- Introduce explicitly documented semantic equivalences.
- Record token usage and inference throughput.
- Compare inference providers and model configurations.
- Introduce repeatable performance benchmarking.