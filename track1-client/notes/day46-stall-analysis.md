# Day 46: CPU Launch Stalls, Host-Device Copies & Synchronization Barriers

## 1. End-to-End Request Latency Decomposition
Total request wall time across an inference step decomposes into four distinct timeline components:

$$T_{\text{total}} = T_{\text{GPU\_compute}} + T_{\text{CPU\_launch\_gap}} + T_{\text{H2D/D2H\_copy}} + T_{\text{sync\_stall}}$$

| Timeline Track | Observed Trace Evidence | Root Cause | Engineering Mitigation |
| :--- | :--- | :--- | :--- |
| **GPU Execution Track** | `gemv2T`, `turing_fp16_gemm`, `fmha_cutlass` kernel bars | Tensor math and VRAM bandwidth reads/writes | Quantization (FP8/INT4), FlashAttention, larger batch size |
| **CPU Launch Track** | White idle gaps between short GPU kernels beneath `cudaLaunchKernel` calls | Python interpreter and PyTorch dispatcher latency exceeding kernel duration | **CUDA Graphs** (capture once, replay entire graph in one launch) |
| **Host-Device Copy Track** | `CUDA memcpy HtoD` / `DtoH` intervals on PCIe bus | Frequent per-step host-device tensor transfers (`.cpu()`, `.to('cuda')`) | Keep KV cache & weights resident on GPU; use `pin_memory=True` + `non_blocking=True` |
| **Synchronization Track** | `cudaStreamSynchronize` / `cudaDeviceSynchronize` blocking intervals | Mid-loop `.item()`, `.cpu()`, or explicit `torch.cuda.synchronize()` calls | Remove mid-loop syncs; perform greedy/top-p token sampling on GPU; check EOS in chunks |

## 2. Asynchronous Launch vs. Stalled Decode Loop Diagnosis
- **Normal Asynchronous Path:** The CPU enqueues `cudaLaunchKernel` commands into the CUDA stream without waiting for completion. Gaps appear only when host dispatch takes longer than a small decode kernel (`CPU launch-bound`).
- **Stalled Sync/Copy Path:** Calling `.cpu()` or `.item()` inside an autoregressive decode loop forces an implicit `cudaStreamSynchronize` plus a PCIe `DtoH` copy. The CPU freezes until all queued kernels finish and data crosses PCIe, leaving the GPU completely idle during the barrier and next launch latency.

## 3. Diagnostic Falsification Rule
- **Rule:** Never attribute idle GPU gaps to slow GPU kernels without inspecting the CPU API, `memcpy`, and synchronization tracks above the gap.
- **Verification:** If `T_GPU_compute` remains constant between a baseline run and a stalled run while `T_total` increases, the regression is strictly caused by host-side synchronization, launch overhead, or PCIe transfer stalls rather than kernel compute degradation.
