# Day 38: Percentile Convention & Statistical Hygiene Notes

## 1. Percentile Definition (`nearest-rank`)
- **Formula:** For $N$ sorted successful measured observations $x_0 \le x_1 \le \dots \le x_{N-1}$ and percentile $q \in (0, 100]$:
  $$\text{index} = \min\left(N - 1,\; \max\left(0,\; \lceil \frac{q}{100} \cdot N \rceil - 1\right)\right)$$
- **Why Nearest-Rank?** Unlike linear interpolation, nearest-rank returns an actual observed request latency. On small sample counts ($N = 5 \text{ to } 20$), tail spikes appear directly in `p95` and `p99` rather than being averaged down.

## 2. Outlier & Failure Policy
- **Outlier Preservation:** High-latency outliers in `status="ok"` requests are **never trimmed or winsorized**.
- **Failure Separation:** Failed requests (`status="error"`, e.g., HTTP 503/timeout) often fail fast. Including them in latency percentiles would artificially improve `p50` while hiding outages. Failures are excluded from latency distributions and reported explicitly via `failure_rate` and `error_breakdown`.
