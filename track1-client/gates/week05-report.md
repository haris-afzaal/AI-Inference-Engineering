# Week 5 Engineering Gate Report: Serving, Batching, PagedAttention & Engine Architecture

**Student:** M.Haris
**Date:** 2026-11-14
**Workspace:** /mnt/b/AI Inference Engineering/track1-client

## 1. Rebuilt Clean Startup Command & Configuration
Command:
`python -m vllm.entrypoints.openai.api_server --model HuggingFaceTB/SmolLM-135M-Instruct --dtype float16 --gpu-memory-utilization 0.85 --max-model-len 2048 --port 8035`

- **Memory & Cache Allocation:** `dtype=float16` cuts weight and KV cache footprint in half vs. `float32`. Profiling allocated `4,120 GPU blocks` and `512 CPU blocks`.
- **Streaming Telemetry:** Verified OpenAI SSE schema (`data: {...}` deltas, `usage` accounting: 24 prompt + 5 completion = 29 tokens, `data: [DONE]`).

## 2. Controlled Failure & Diagnostic Layer Matrix

| Test Case | Symptom | System Layer | Root Cause & Fix |
| :--- | :--- | :--- | :--- |
| **Static vs. Paged Memory (Day 29)** | 90.25% VRAM wasted on short prompts | KV Cache Memory Allocator | Pre-allocating `max_model_len` contiguously causes internal fragmentation; PagedAttention 16-token blocks cut waste by 275.9x. |
| **Over-Allocation (`gpu_util=1.1`, Day 30)** | Immediate startup crash | Engine Config Validator | Utilization exceeded `1.0`; capped `gpu_memory_utilization <= 0.90` to preserve activation headroom. |
| **Missing Triton Config (Day 32)** | `[ERROR] Failed to load 'whisper_model'` | Triton Model Repository Manager | Triton enforces strict tensor schemas; resolved by providing `config.pbtxt` + version directory (`1/`). |
| **Context Limit Breach (`max_tokens=4096`, Day 35)** | `HTTP 400 context_length_exceeded` | API Validation / Admission Guard | Requested `4,120` tokens > `max_model_len=2048`. Prevented OOM before prefill; fixed by clamping `max_tokens <= (max_model_len - prompt_tokens)`. |

## 3. Five-Minute Engine Choice Defense (vLLM vs. Triton vs. SGLang)

1. **Choose vLLM (Default LLM Production Workhorse):** PagedAttention eliminates KV cache fragmentation, continuous batching maximizes GPU throughput, and its native OpenAI-compatible HTTP/SSE server provides drop-in compatibility. Trade-off: single-model per instance.
2. **Choose NVIDIA Triton Inference Server (Multi-Model / Heterogeneous Pipelines):** Unified enterprise orchestrator capable of serving non-LLM models (Whisper audio, vision encoders) alongside vLLM on one GPU with strict tensor routing (`config.pbtxt`). Trade-off: rigid setup overhead.
3. **Choose SGLang (Prefix-Heavy RAG & Multi-Turn Agents):** RadixAttention maintains a Radix Tree of KV cache blocks, automatically sharing and reusing common system prompts or long RAG contexts across concurrent requests. Trade-off: CPU tree-lookup overhead when prompts share zero prefix.
