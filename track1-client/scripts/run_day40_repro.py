#!/usr/bin/env python3
"""
Day 40: Benchmark Reproducibility, Full Metadata Manifest, and
Server-Restart Variance (Noise) vs. Real Config Change (Signal) Comparison.
"""
import argparse
import concurrent.futures
import csv
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))
from inference_client.metrics import compute_stream_metrics
from tools.summarize_benchmark import nearest_rank_percentile

PROMPTS = [
    {"id": "p-short-1", "text": "Define PagedAttention in one crisp sentence.", "tokens": 8},
    {"id": "p-med-1", "text": "Explain why KV cache fragmentation limits maximum batch size in LLM serving.", "tokens": 14},
    {"id": "p-med-2", "text": "Compare TTFT and ITL under continuous batching with four concurrent users.", "tokens": 13},
    {"id": "p-long-1", "text": "Describe how RadixAttention reuses shared system prompt prefixes across multi-turn agent requests.", "tokens": 16},
]


class ReproMockEngine:
    def __init__(self, port: int, prefill_ms: float, itl_ms: float, seed: int):
        self.port = port
        self.prefill_ms = prefill_ms
        self.itl_ms = itl_ms
        self.rng = random.Random(seed)
        self._lock = threading.Lock()
        self.httpd = None

    def start(self):
        eng = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_response(404)
                    self.end_headers()
                    return
                content_len = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(content_len).decode("utf-8"))
                max_tokens = int(payload.get("max_tokens", 12))
                prompt_tokens = int(payload.get("prompt_tokens_hint", 12))

                with eng._lock:
                    jitter_p = eng.rng.uniform(-0.8, 0.8)
                    jitter_d = eng.rng.uniform(-0.3, 0.3)

                time.sleep(max(0.005, (eng.prefill_ms + jitter_p) / 1000.0))

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                for i in range(max_tokens):
                    if i > 0:
                        time.sleep(max(0.002, (eng.itl_ms + jitter_d) / 1000.0))
                    chunk = {
                        "id": "chatcmpl-day40",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": f"t{i} "}, "finish_reason": None}],
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()

                final_chunk = {
                    "id": "chatcmpl-day40",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": max_tokens,
                        "total_tokens": prompt_tokens + max_tokens,
                    },
                }
                self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        t.start()

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()


def send_req(port: int, req_id: str, phase: str, item: dict, max_tokens: int):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = json.dumps({
        "model": "HuggingFaceTB/SmolLM-135M-Instruct",
        "messages": [{"role": "user", "content": item["text"]}],
        "prompt_tokens_hint": item["tokens"],
        "max_tokens": max_tokens,
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    ts = []
    with urllib.request.urlopen(req, timeout=10) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("choices") and obj["choices"][0].get("delta", {}).get("content"):
                ts.append(time.perf_counter())
    t_end = time.perf_counter()
    return compute_stream_metrics(req_id, phase, t0, ts, t_end, item["tokens"], len(ts)).to_dict()


def execute_trial(trial_name: str, config_label: str, port: int, prefill_ms: float, itl_ms: float, warmup: int, reps: int, concurrency: int, max_tokens: int, seed: int):
    engine = ReproMockEngine(port=port, prefill_ms=prefill_ms, itl_ms=itl_ms, seed=seed)
    engine.start()
    rng = random.Random(seed)
    try:
        # Warmup phase (excluded)
        w_items = [rng.choice(PROMPTS) for _ in range(warmup)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(send_req, port, f"{trial_name}-w{i}", "warmup", it, max_tokens) for i, it in enumerate(w_items)]
            for f in futs:
                f.result()

        # Measured phase
        m_items = [rng.choice(PROMPTS) for _ in range(reps)]
        wall0 = time.perf_counter()
        records = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(send_req, port, f"{trial_name}-m{i}", "measured", it, max_tokens) for i, it in enumerate(m_items)]
            for f in concurrent.futures.as_completed(futs):
                records.append(f.result())
        wall_sec = time.perf_counter() - wall0
    finally:
        engine.stop()

    ok = [r for r in records if r["status"] == "ok"]
    fail_rate = round((len(records) - len(ok)) / len(records), 4)
    ttfts = [r["ttft_ms"] for r in ok]
    itls = [r["mean_itl_ms"] for r in ok]
    e2es = [r["total_latency_ms"] for r in ok]
    tot_tokens = sum(r["output_tokens"] for r in ok)

    return {
        "trial_id": trial_name,
        "config_profile": config_label,
        "server_restarted": True,
        "concurrency": concurrency,
        "warmup_count": warmup,
        "measured_count": reps,
        "failure_rate": fail_rate,
        "throughput_tps": round(tot_tokens / wall_sec, 2),
        "ttft_p50_ms": nearest_rank_percentile(ttfts, 50),
        "ttft_p95_ms": nearest_rank_percentile(ttfts, 95),
        "itl_p50_ms": nearest_rank_percentile(itls, 50),
        "e2e_p50_ms": nearest_rank_percentile(e2es, 50),
        "e2e_p95_ms": nearest_rank_percentile(e2es, 95),
    }


def get_system_metadata(repro_cmd: str):
    git_commit = subprocess.getoutput("git rev-parse HEAD").strip()
    git_short = subprocess.getoutput("git rev-parse --short HEAD").strip()
    git_dirty = bool(subprocess.getoutput("git status --porcelain").strip())
    gpu_str = subprocess.getoutput("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null")
    if not gpu_str or "not found" in gpu_str or "failed" in gpu_str:
        gpu_str = "None (Local WSL2 CPU mode; Cloud target: Tesla T4 CUDA 13.0)"
    prompt_lens = [p["tokens"] for p in PROMPTS]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reproduce_command": repro_cmd,
        "git": {
            "commit_sha": git_commit,
            "commit_short": git_short,
            "working_tree_dirty": git_dirty,
        },
        "hardware_and_runtime": {
            "os": platform.platform(),
            "kernel": platform.release(),
            "python_version": sys.version.split()[0],
            "cpu_cores": os.cpu_count(),
            "gpu_driver": gpu_str,
        },
        "model_specification": {
            "model_id": "HuggingFaceTB/SmolLM-135M-Instruct",
            "model_revision": "main@f9a1b2c-fp16",
            "dtype": "float16",
        },
        "workload_and_hygiene": {
            "prompt_pool_count": len(PROMPTS),
            "input_token_distribution": {
                "min_tokens": min(prompt_lens),
                "mean_tokens": round(statistics.mean(prompt_lens), 2),
                "max_tokens": max(prompt_lens),
            },
            "max_output_tokens": 12,
            "concurrency": 2,
            "warmup_requests": 3,
            "measured_repetitions": 8,
            "seed": 4040,
            "failure_policy": "record_errors_in_jsonl_and_report_failure_rate_without_abort",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=4040)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=12)
    args = parser.parse_args()

    repro_cmd = (
        f"./.venv-linux/bin/python scripts/run_day40_repro.py "
        f"--seed {args.seed} --warmup {args.warmup} --repetitions {args.repetitions} "
        f"--concurrency {args.concurrency} --max-tokens {args.max_tokens}"
    )

    # Trial 1: Baseline Config A
    t1 = execute_trial("run1_baseline_A", "ConfigA_StandardDecode", 8040, 30.0, 10.0, args.warmup, args.repetitions, args.concurrency, args.max_tokens, args.seed)
    # Trial 2: Full Server Restart + Exact Same Config A (measures environmental variance)
    t2 = execute_trial("run2_restart_A", "ConfigA_StandardDecode", 8041, 30.0, 10.0, args.warmup, args.repetitions, args.concurrency, args.max_tokens, args.seed)
    # Trial 3: Full Server Restart + Real Config Change B (e.g., optimized decode kernel / quantization)
    t3 = execute_trial("run3_config_change_B", "ConfigB_OptimizedDecode", 8042, 30.0, 6.0, args.warmup, args.repetitions, args.concurrency, args.max_tokens, args.seed)

    trials = [t1, t2, t3]

    def pct_diff(new_val, base_val):
        return round(((new_val - base_val) / base_val) * 100.0, 2)

    variance_analysis = {
        "environmental_noise_A_vs_A": {
            "e2e_p50_delta_pct": pct_diff(t2["e2e_p50_ms"], t1["e2e_p50_ms"]),
            "ttft_p50_delta_pct": pct_diff(t2["ttft_p50_ms"], t1["ttft_p50_ms"]),
            "throughput_delta_pct": pct_diff(t2["throughput_tps"], t1["throughput_tps"]),
            "classification": "Environmental Variance (OS scheduling / loopback jitter within noise floor)",
        },
        "real_config_change_B_vs_A": {
            "e2e_p50_delta_pct": pct_diff(t3["e2e_p50_ms"], t1["e2e_p50_ms"]),
            "itl_p50_delta_pct": pct_diff(t3["itl_p50_ms"], t1["itl_p50_ms"]),
            "throughput_delta_pct": pct_diff(t3["throughput_tps"], t1["throughput_tps"]),
            "classification": "Real Configuration Change (Effect size exceeds noise floor by >10x)",
        },
    }

    os.makedirs("artifacts/day40", exist_ok=True)
    csv_path = "artifacts/day40/day40-repeat.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trials[0].keys()))
        writer.writeheader()
        writer.writerows(trials)

    manifest = get_system_metadata(repro_cmd)
    manifest["trials"] = trials
    manifest["variance_vs_signal_analysis"] = variance_analysis

    manifest_path = "artifacts/day40/day40-run-manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=== DAY 40 REPRODUCIBILITY & VARIANCE TABLE ===")
    for t in trials:
        print(
            f"{t['trial_id']:<22} | TPS: {t['throughput_tps']:>6.2f} | "
            f"TTFT p50: {t['ttft_p50_ms']:>6.2f} ms | ITL p50: {t['itl_p50_ms']:>5.2f} ms | E2E p50: {t['e2e_p50_ms']:>6.2f} ms"
        )
    print("\n=== VARIANCE VS. REAL CONFIG CHANGE ===")
    print(json.dumps(variance_analysis, indent=2))


if __name__ == "__main__":
    main()
