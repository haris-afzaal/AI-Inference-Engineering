# Week 07 Gate Report: Profiling, Memory Accounting, Roofline & Bounded Tuning

## 1. Bottleneck Defense (Memory, Roofline, Benchmark & Timeline Evidence)
* **Memory Accounting (`notes/day43-memory.md`):** Switching model weights from `float32` (`4 bytes/param`) to `float16` (`2 bytes/param`) reduced static weight footprint by exactly `50.0%` (`192.2 MB` $\rightarrow$ `96.1 MB`), increasing KV cache headroom (`available_vram - (weights + kv_cache + workspace)`).
* **Timeline & Stall Attribution (`notes/day46-stall-analysis.md`):** On Tesla T4 Nsight Systems traces, a `68.72 ms` request spent only `9.37 ms` (`13.6%`) in active GPU kernels (`gemv2T_kernel_val`, `turing_fp16_gemm`) while `59.35 ms` (`86.4%`) was spent in host Python launch dispatch, PCIe transfers, and `cudaStreamSynchronize` barriers. Idle GPU gaps must not be blamed on slow GPU kernels without checking sync and copy tracks.
* **Bounded Tuning & Generalization (`notes/day47-experiment.md`, `notes/day48-experiment-2.md`):**
  - On short sequences (`seq_len=128`), `float16` improved local CPU `p50` latency by `-11.57%` (`408.66 ms` $\rightarrow$ `361.36 ms`) with `top1_match=True` and `0 NaNs`, exceeding the $\pm 3\%$ noise floor.
  - Across longer sequences (`seq_len=512`), `float16` regressed local CPU latency by `+220.42%` (`609.68 ms` $\rightarrow$ `1953.52 ms`) due to missing native FP16 matrix instructions (`FP16 <-> FP32` fallback overhead), proving that tuning gains are hardware- and regime-dependent.

## 2. Misleading Chart Label Repair & Bounded Assumption Failure
* **Misleading Label Diagnosed:** *"GPU Compute Bottleneck: 86.4% Slow GPU Kernels"* paired with a hardcoded `200 ms` Y-axis ceiling.
* **Bounded Assumption Tested:** Assumed a fixed `200 ms` Y-axis ceiling covers all latency comparisons; this clipped both `seq_len=512` bars (`609.68 ms` and `1953.52 ms`), hiding the `3.2x` regression.
* **Repair Applied (`artifacts/day49/day49-corrected-chart.svg`):** Replaced with dynamic zero-based scaling (`0–2000 ms`), explicit `p50` labels, and accurate separation of `13.6%` active GPU kernel time vs. `86.4%` host dispatch/sync/copy overhead.

## 3. What Was Not Measured (Explicit Blind Spots)
1. **Python-Level Host Call Attribution:** Which exact Python framework functions consumed the `59.35 ms` host gap (requires `cProfile` or `torch.profiler` stack traces).
2. **Multi-Tenant Dynamic Batching Contention:** Queueing delay and KV cache fragmentation under concurrent production traffic (`vLLM` / `TensorRT-LLM` continuous batching).
3. **Sustained Multi-Stream PCIe Saturation:** Concurrent `H2D`/`D2H` weight swapping under multi-request load.

## 4. Week 07 Gate Verdict
* **Status:** **PASSED** — Hypothesis supported with quantitative memory, roofline, timeline, and A/B tuning evidence; misleading chart repaired and unmeasured blind spots explicitly documented.
