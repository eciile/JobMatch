# JobMatch — LLM Evaluation

## Objective

Evaluate LLM-based requirements extraction and compare inference providers, models and generation configurations independently of webpage retrieval and candidate matching.

## Dataset

`datasets/job_requirements.jsonl` contains six synthetic French and English job advertisements with manually annotated required skills, preferred skills, qualifications, soft skills and languages.

## Usage

From the repository root, with the configured inference server running:

```bash
python -m evaluation.evaluate_extraction
```

The evaluation runner uses the application's configured inference provider.

## Metrics

- Micro-averaged precision, recall and F1-score.
- Per-category extraction performance.
- Extraction success and exact match rates.
- End-to-end inference latency.
- False positives and false negatives.

## Results

Model: `qwen3:4b-instruct-2507-q4_K_M`  
Hardware: NVIDIA GeForce RTX 5060 (8 GB VRAM).

| Experiment | Precision | Recall | F1 | Mean latency |
|---|---:|---:|---:|---:|
| V0 — Initial baseline | 57.14% | 71.43% | 63.49% | 2.98 s |
| V1 — Extended prompt | 52.78% | 67.86% | 59.37% | 3.33 s |
| V2 — Revised prompt | 60.61% | 71.43% | 65.57% | 2.43 s |
| Modular Ollama — Cold start | 57.58% | 67.86% | 62.30% | 4.17 s |
| Modular Ollama — Warm | 57.58% | 67.86% | 62.30% | 1.54 s |

All recorded runs achieved a 100% extraction success rate.

Detailed configurations and results are stored in [`results/`](results/).

## Limitations

- The small synthetic dataset is not representative of real-world extraction accuracy.
- Exact matching may penalize semantically equivalent expressions.
- Successful extraction does not guarantee semantic correctness.
- Latency depends on model-loading conditions and runtime configuration.

## Future Work

- Compare Ollama, llama.cpp and vLLM.
- Measure throughput, time to first token, VRAM usage and concurrency.
- Establish reproducible cold-start and warm inference benchmarks.
- Expand the evaluation dataset.

See the [Inference Architecture](../docs/inference-architecture.md) for implementation details.