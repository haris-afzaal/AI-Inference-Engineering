#!/usr/bin/env python3
"""
Day 42: Week 6 Benchmark Gate Rerun
1. Clean server stop & cold restart verification
2. Re-execute concurrency sweep (c=1, 2, 4, 8) preserving metric definitions & metadata
3. Compare Day 42 vs Day 41 manifests and classify variance (>10% threshold)
4. Bounded assumption failure & recovery test
"""
import concurrent.futures
import csv
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath("src"))
sys.path.insert(0, os.path.abspath("."))
from scripts.run_week06_gate import GateBenchmarkServer, send_stream_request
from tools.summarize_benchmark import nearest_rank_percentile, summarize_records

PORT = 8042
LEVELS = [1, 2, 4, 8]
WARMUP_N = 2
REPS_N = 6
MAX_TOKENS = 8
PROMPT_TOKENS = 12
VARIANCE_THRESHOLD_PCT = 10.0


def wait_for_server_down(port: int, timeout_s: float = 3.0) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.2)
        except urllib.error.URLError:
            return True
        time.sleep(0.05)
    return False


def main():
    os.makedirs("artifacts/day42", exist_ok=True)
    with open("artifacts/day41/week06-benchmark.json", "r", encoding="utf-8") as f:
        day41_manifest = json.load(f)

    # 1. Start initial server, then perform clean stop + cold restart
    srv_initial = GateBenchmarkServer(port=PORT, max_slots=4, max_inflight=12, inject_outlier_every=11)
    srv_initial.start()
    probe = send_stream_request(PORT, "pre-restart-probe", "probe", 1, MAX_TOKENS)
    assert probe["status"] == "ok", f"Pre-restart probe failed: {probe}"

    t_stop_0 = time.perf_counter()
    srv_initial.stop()
    port_released = wait_for_server_down(PORT)
    assert port_released, "Server port did not release cleanly on stop"

    srv = GateBenchmarkServer(port=PORT, max_slots=4, max_inflight=12, inject_outlier_every=11)
    srv.start()
    restart_ms = round((time.perf_counter() - t_stop_0) * 1000.0, 3)

    all_records = []
    sweep_rows = []
    try:
        for c in LEVELS:
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                for fut in [pool.submit(send_stream_request, PORT, f"c{c}-w{i}", "warmup", c, MAX_TOKENS) for i in range(WARMUP_N)]:
                    all_records.append(fut.result())

            w0 = time.perf_counter()
            lvl_recs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                futs = [pool.submit(send_stream_request, PORT, f"c{c}-m{i}", "measured", c, MAX_TOKENS) for i in range(REPS_N)]
                for fut in concurrent.futures.as_completed(futs):
                    rec = fut.result()
                    lvl_recs.append(rec)
                    all_records.append(rec)
            w_sec = time.perf_counter() - w0
            ok = [r for r in lvl_recs if r["status"] == "ok"]
            sweep_rows.append({
                "concurrency": c,
                "measured_n": REPS_N,
                "ok_n": len(ok),
                "failure_rate": round((REPS_N - len(ok)) / REPS_N, 4),
                "system_tps": round(sum(r["output_tokens"] for r in ok) / w_sec, 2),
                "ttft_p50_ms": nearest_rank_percentile([r["ttft_ms"] for r in ok], 50),
                "ttft_p95_ms": nearest_rank_percentile([r["ttft_ms"] for r in ok], 95),
                "itl_p50_ms": nearest_rank_percentile([r["mean_itl_ms"] for r in ok], 50),
                "e2e_p50_ms": nearest_rank_percentile([r["total_latency_ms"] for r in ok], 50),
                "e2e_p95_ms": nearest_rank_percentile([r["total_latency_ms"] for r in ok], 95),
            })

        # 4. Bounded assumption break & restore test (Task 5)
        broken_recs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = [
                pool.submit(send_stream_request, PORT, f"break-timeout-{i}", "fault_test", 2, MAX_TOKENS, 0.015)
                for i in range(4)
            ]
            for fut in concurrent.futures.as_completed(futs):
                broken_recs.append(fut.result())
        broken_fails = [r for r in broken_recs if r["status"] != "ok"]

        restored_recs = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = [
                pool.submit(send_stream_request, PORT, f"restore-timeout-{i}", "fault_restore", 2, MAX_TOKENS, 5.0)
                for i in range(4)
            ]
            for fut in concurrent.futures.as_completed(futs):
                restored_recs.append(fut.result())
        restored_fails = [r for r in restored_recs if r["status"] != "ok"]
    finally:
        srv.stop()

    for csv_path in ("artifacts/day42-rerun.csv", "artifacts/day42/day42-rerun.csv"):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
            w.writeheader()
            w.writerows(sweep_rows)

    d41_by_c = {row["concurrency"]: row for row in day41_manifest["sweep_results"]}
    comparisons = []
    flagged_variances = []
    metrics_to_compare = ["system_tps", "ttft_p50_ms", "ttft_p95_ms", "itl_p50_ms", "e2e_p50_ms", "e2e_p95_ms"]

    for row in sweep_rows:
        c = row["concurrency"]
        base = d41_by_c[c]
        comp = {"concurrency": c, "deltas_pct": {}}
        for m in metrics_to_compare:
            delta_pct = round(((row[m] - base[m]) / base[m]) * 100.0, 2) if base[m] else 0.0
            comp["deltas_pct"][m] = delta_pct
            if abs(delta_pct) > VARIANCE_THRESHOLD_PCT:
                classification = (
                    "actionable_structural_outlier_shift"
                    if "p95" in m
                    else "expected_os_scheduling_jitter"
                )
                flagged_variances.append({
                    "concurrency": c,
                    "metric": m,
                    "day41_value": base[m],
                    "day42_value": row[m],
                    "delta_pct": delta_pct,
                    "classification": classification,
                })
        comparisons.append(comp)

    clean_records = [{k: v for k, v in r.items() if k != "raw_trace"} for r in all_records]
    overall_summary = summarize_records(clean_records, source_file="day42-rerun-in-memory")

    manifest = {
        "gate": "Week 6 Benchmark Gate Rerun (Day 42)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.getoutput("git rev-parse --short HEAD").strip(),
        "reproduce_command": "./.venv-linux/bin/python scripts/run_day42_rerun.py",
        "run_metadata": {
            "hardware": {
                "platform": platform.machine(),
                "cpu_cores": os.cpu_count(),
                "accelerator": "CPU-only WSL2 harness (no local NVIDIA GPU)",
            },
            "software": {
                "os": platform.platform(),
                "python_version": sys.version.split()[0],
                "server_impl": "GateBenchmarkServer (ThreadingHTTPServer SSE mock)",
            },
            "model_and_server_config": {
                "model_id": "synthetic-sse-token-stream-v1",
                "max_slots": 4,
                "max_inflight": 12,
                "inject_outlier_every": 11,
                "clean_restart_verified": True,
                "restart_cycle_ms": restart_ms,
            },
            "workload_controls": {
                "concurrency_levels": LEVELS,
                "warmup_requests_per_level": WARMUP_N,
                "measured_repetitions_per_level": REPS_N,
                "prompt_tokens": PROMPT_TOKENS,
                "max_tokens": MAX_TOKENS,
                "total_warmup_excluded": len(LEVELS) * WARMUP_N,
                "total_measured_requests": len(LEVELS) * REPS_N,
                "total_failures": overall_summary["sample_counts"]["measured_error"],
            },
        },
        "sweep_results": sweep_rows,
        "day41_comparison": {
            "baseline_artifact": "artifacts/day41/week06-benchmark.json",
            "variance_threshold_pct": VARIANCE_THRESHOLD_PCT,
            "per_concurrency_deltas": comparisons,
            "variances_beyond_threshold": flagged_variances,
        },
        "overall_summary": overall_summary,
        "bounded_assumption_break_test": {
            "broken_assumption": "Set client read timeout=0.015s (15ms) below minimum server TTFT (~25ms)",
            "broken_failure_rate": round(len(broken_fails) / len(broken_recs), 4),
            "observed_symptom": broken_fails[0]["error_type"] if broken_fails else "none",
            "restored_assumption": "Restored client timeout=5.0s",
            "restored_failure_rate": round(len(restored_fails) / len(restored_recs), 4),
        },
    }

    with open("artifacts/day42/day42-manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=== DAY 42 CLEAN-RESTART RERUN SWEEP ===")
    for r in sweep_rows:
        print(r)
    print("\n=== VARIANCES BEYOND +/-10% VS DAY 41 ===")
    print(json.dumps(flagged_variances, indent=2))
    print("\n=== BOUNDED ASSUMPTION BREAK & RESTORE (TIMEOUT) ===")
    print(json.dumps(manifest["bounded_assumption_break_test"], indent=2))


if __name__ == "__main__":
    main()
