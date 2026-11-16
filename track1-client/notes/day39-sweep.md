# Day 39: Concurrency Sweep & Saturation Analysis

## 1. Experimental Setup & Statistical Hygiene
- **Engine Configuration:** `HuggingFaceTB/SmolLM-135M-Instruct` (`max_tokens=16`, `MAX_ACTIVE_SLOTS=4`, `MAX_TOTAL_INFLIGHT=8`).
- **Sweep Protocol:** Concurrency levels `[1, 2, 4, 8, 12]` with `2` warm-up requests (excluded) and `5` measured repetitions per level.
- **Artifacts:** `artifacts/day39/day39-concurrency.csv`, `artifacts/day39/latency_vs_throughput.svg`, `artifacts/day39/manifest.json`.

## 2. Measured Results Summary

| Concurrency | OK / N | Failure Rate | System Throughput (tok/s) | TTFT p50 (ms) | TTFT p95 (ms) | ITL p50 (ms) | E2E p50 (ms) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 5 / 5 | 0.0% | 97.36 | 33.55 | 35.31 | 8.60 | 163.65 |
| 2 | 5 / 5 | 0.0% | 140.17 | 36.42 | 38.61 | 11.05 | 202.20 |
| 4 | 5 / 5 | 0.0% | 176.31 | 42.73 | 51.87 | 15.94 | 279.05 |
| 8 | 5 / 5 | 0.0% | 182.85 | 47.95 | 96.81 | 15.96 | 287.63 |
| 12 | 5 / 5 | 0.0% | 182.97 | 52.11 | 95.95 | 16.04 | 292.43 |

## 3. Saturation Knee Identification (`Concurrency = 4`)
The optimal operating knee is **`concurrency = 4`**, where aggregate throughput reaches **96.4% of peak capacity** (`176.31 tok/s` vs. `182.97 tok/s` max) before severe tail queueing begins. Pushing concurrency beyond `4` triggers clear saturation across three measured signals:

1. **Throughput Plateau:** Scaling from `c=1` to `c=4` yields `+81.1%` throughput (`97.36 -> 176.31 tok/s`), whereas scaling from `c=4` to `c=12` yields only `+3.8%` (`176.31 -> 182.97 tok/s`), proving compute/batch slot saturation.
2. **Tail TTFT Spike (`p95`):** `TTFT p95` jumps `+86.6%` between `c=4` (`51.87 ms`) and `c=8` (`96.81 ms`) because excess arrivals queue behind active slots before prefill can start.
3. **ITL & End-to-End Degradation:** Memory-bandwidth contention across concurrent decode streams pushes `ITL p50` from `8.60 ms` (`c=1`) to `16.04 ms` (`c=12`), inflating `E2E p50` latency by `+78.7%` (`163.65 -> 292.43 ms`).
