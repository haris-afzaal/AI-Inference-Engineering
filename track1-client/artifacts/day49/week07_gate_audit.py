import json
import math

# 1. Consolidated Week 7 Evidence (Days 43-48)
evidence = {
    "hardware_telemetry": {
        "gpu_target": "NVIDIA Tesla T4 (15360 MiB VRAM, FP16 Tensor Core 65 TFLOPS, 320 GB/s HBM)",
        "local_cpu_target": "WSL2 Ubuntu x86 CPU (No FP16 Tensor Cores; FP16->FP32 fallback on long prefill)"
    },
    "memory_accounting": {
        "model_params": "50.38M (4-layer Decoder, d_model=1024)",
        "fp32_weights_mb": 192.2,
        "fp16_weights_mb": 96.1,
        "memory_reduction_pct": -50.0
    },
    "timeline_breakdown_t4_ms": {
        "total_request_nvtx_ms": 68.718,
        "gpu_kernel_compute_ms": 9.370,
        "cpu_launch_and_python_dispatch_ms": 55.148,
        "h2d_d2h_memcpy_ms": 2.799,
        "sync_barrier_overhead_ms": 1.401,
        "gpu_active_pct": 13.64
    },
    "tuning_experiments": {
        "day47_t4_dtype_fp16": {"delta_latency_pct": -40.0, "delta_mem_pct": -50.0, "verdict": "IMPROVE"},
        "day48_cpu_seq128_fp16": {"delta_latency_pct": -11.57, "delta_mem_pct": -50.0, "verdict": "IMPROVE"},
        "day48_cpu_seq512_fp16": {"delta_latency_pct": 220.42, "delta_mem_pct": -50.0, "verdict": "REGRESS (DOES NOT GENERALIZE)"}
    },
    "what_was_not_measured": [
        "Python interpreter vs PyTorch dispatcher breakdown inside the 55.15 ms host gap (requires torch.profiler / cProfile)",
        "CUDA Graph capture/replay latency reduction on Tesla T4 decode loop",
        "Multi-user concurrent batching (concurrency > 1) KV cache fragmentation under continuous batching",
        "Real LLM weights (e.g., Llama-3-8B) end-to-end TTFT/ITL under INT4/FP8 quantization"
    ]
}

# 2. Misleading Chart Audit & Repair
misleading_chart = {
    "title": "GPU Kernel Optimization Makes Inference 86% Faster Everywhere!",
    "x_axis": "Optimization",
    "y_axis": "Speed (no units, truncated axis 55ms-70ms)",
    "flaws": [
        "False attribution: Labels host sync/launch reduction as 'GPU Kernel Optimization' when GPU kernels were only 13.64% of runtime.",
        "Missing units & truncated Y-axis: Exaggerates visual bar difference by starting Y-axis above zero.",
        "Over-generalization ('Everywhere!'): Hides Day 48 seq_len=512 CPU regression (+220.42% slower)."
    ]
}

# 3. Generate Corrected, Honest SVG Chart (Zero-based Y-axis, explicit ms units, exact scope)
bars = [
    ("T4 FP32 Baseline (seq=128)", 100.0, "#4A90E2"),
    ("T4 FP16 Candidate (seq=128)", 60.0, "#2ECC71"),
    ("CPU FP32 Baseline (seq=512)", 609.7, "#4A90E2"),
    ("CPU FP16 Candidate (seq=512)", 1953.5, "#E74C3C"),
]

# Bounded Assumption Break Test: Fixed-scale chart (max_y=200ms) clips 1953.5ms bar!
fixed_max_y = 200.0
clipped_bars = [label for label, val, _ in bars if val > fixed_max_y]
assert len(clipped_bars) == 2, "Expected clipping under naive fixed-scale assumption!"

# Dynamic zero-based scaling repair
max_y = 2000.0
chart_h = 260
svg_lines = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="860" height="420" font-family="Arial, sans-serif">',
    '  <rect width="100%" height="100%" fill="#ffffff"/>',
    '  <text x="430" y="32" text-anchor="middle" font-size="16" font-weight="bold" fill="#111">Corrected: End-to-End Request Latency (p50 ms, Lower is Better) by Hardware &amp; Prompt Length</text>',
    '  <text x="430" y="54" text-anchor="middle" font-size="12" fill="#555">Zero-based Y-axis (0–2000 ms) | 4-Layer Decoder (d_model=1024, 12 decode steps, n=5 reps)</text>',
    '  <line x1="90" y1="330" x2="810" y2="330" stroke="#333" stroke-width="2"/>',
    '  <line x1="90" y1="70" x2="90" y2="330" stroke="#333" stroke-width="2"/>',
    '  <text x="25" y="200" transform="rotate(-90 25,200)" text-anchor="middle" font-size="13" font-weight="bold" fill="#222">p50 Latency (ms)</text>',
]

for tick in [0, 500, 1000, 1500, 2000]:
    y_pos = 330 - int((tick / max_y) * chart_h)
    svg_lines.append(f'  <line x1="85" y1="{y_pos}" x2="810" y2="{y_pos}" stroke="#e0e0e0" stroke-dasharray="4"/>')
    svg_lines.append(f'  <text x="80" y="{y_pos + 4}" text-anchor="end" font-size="11" fill="#444">{tick} ms</text>')

for idx, (label, val, color) in enumerate(bars):
    x_pos = 130 + idx * 170
    bar_h = max(4, int((val / max_y) * chart_h))
    y_pos = 330 - bar_h
    svg_lines.append(f'  <rect x="{x_pos}" y="{y_pos}" width="95" height="{bar_h}" fill="{color}" rx="4"/>')
    svg_lines.append(f'  <text x="{x_pos + 47}" y="{y_pos - 8}" text-anchor="middle" font-size="12" font-weight="bold" fill="#111">{val:.1f} ms</text>')
    svg_lines.append(f'  <text x="{x_pos + 47}" y="352" text-anchor="middle" font-size="11" fill="#222">{label.split(" (")[0]}</text>')
    svg_lines.append(f'  <text x="{x_pos + 47}" y="368" text-anchor="middle" font-size="11" fill="#555">({label.split(" (")[1]}</text>')

svg_lines.append('  <text x="430" y="402" text-anchor="middle" font-size="11" fill="#333">Note: FP16 improves latency on T4 Tensor Cores (-40%), but regresses on x86 CPU at seq_len=512 (+220.4%).</text>')
svg_lines.append('</svg>')

with open("artifacts/day49/day49-corrected-chart.svg", "w") as f:
    f.write("\n".join(svg_lines))

summary = {
    "gate": "Week 07 Gate (Days 43-49)",
    "misleading_chart_audit": misleading_chart,
    "bounded_assumption_break": {
        "assumption": "Hardcoded Y-axis ceiling of 200 ms is sufficient for all latency comparisons",
        "failure_symptom": f"Clipped bars exceeding 200 ms: {clipped_bars}",
        "fix": "Dynamic zero-based Y-axis scaling (0 to 2000 ms) with explicit p50 ms labels"
    },
    "evidence": evidence
}

with open("artifacts/day49/day49-final-summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("Verified Week 07 Gate Audit:")
print(" - Clipped bars detected under fixed-scale assumption:", clipped_bars)
print(" - Generated artifacts/day49/day49-corrected-chart.svg")
print(" - Generated artifacts/day49/day49-final-summary.json")
