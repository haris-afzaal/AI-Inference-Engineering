# Day 40: Benchmark Reproducibility & Environmental Variance Analysis

## 1. Controlled Reproducibility Contract
Every benchmark execution records a self-contained manifest (`artifacts/day40/day40-run-manifest.json`) capturing:
- **Code & Model Provenance:** Git commit SHA (`cab9e6b`), dirty tree flag, model ID (`HuggingFaceTB/SmolLM-135M-Instruct`), revision, and `float16` precision.
- **Hardware & Runtime:** WSL2 Linux kernel, CPU core count, Python runtime (`3.14.4`), and target GPU/driver metadata.
- **Workload & Statistical Hygiene:** Prompt token distribution (`min=8, mean=12.75, max=16`), `max_tokens=12`, `concurrency=2`, `warmup=3` (excluded), `repetitions=8`, deterministic `seed=4040`, and non-aborting JSONL failure policy.
- **Exact Reproduction Command:**
  ```bash
  ./.venv-linux/bin/python scripts/run_day40_repro.py --seed 4040 --warmup 3 --repetitions 8 --concurrency 2 --max-tokens 12
