#!/usr/bin/env python3
"""
Day 41: Week 6 Gate Benchmark Suite
1. Clean concurrency sweep (c=1, 2, 4, 8) with full reproducibility manifest
2. Independent hand-formula verification on a single raw trace
3. Tail outlier identification (nearest-rank p95 / max)
4. Bounded assumption break (admission queue cap = 1) -> failure symptom -> restore
"""
import concurrent.futures
import csv
import json
import os
import platform
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
from tools.summarize_benchmark import nearest_rank_percentile, summarize_records

PORT = 8041


class GateBenchmarkServer:
    def __init__(self, port: int, max_slots: int = 4, max_inflight: int = 12, inject_outlier_every: int = 11):
        self.port = port
        self.max_slots = max_slots
        self.max_inflight = max_inflight
        self.inject_outlier_every = inject_outlier_every
        self._lock = threading.Lock()
        self._inflight = 0
        self._req_counter = 0
        self.httpd = None

    def start(self):
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_response(404)
                    self.end_headers()
                    return

                with srv._lock:
                    if srv._inflight >= srv.max_inflight:
                        err = json.dumps({
                            "error": {
                                "message": f"Admission cap reached: inflight={srv._inflight} >= max_inflight={srv.max_inflight}",
                                "type": "capacity_exceeded",
                                "code": "queue_full_503",
                            }
                        }).encode("utf-8")
                        self.send_response(503)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(err)))
                        self.end_headers()
                        self.wfile.write(err)
                        return
                    srv._inflight += 1
                    srv._req_counter += 1
                    load = srv._inflight
                    req_seq = srv._req_counter

                try:
                    content_len = int(self.headers.get("Content-Length", 0))
                    payload = json.loads(self.rfile.read(content_len).decode("utf-8"))
                    max_tokens = int(payload.get("max_tokens", 8))
                    prompt_tokens = int(payload.get("prompt_tokens", 12))

                    queue_wait = max(0, load - srv.max_slots) * 0.035
                    outlier_pause = 0.140 if (srv.inject_outlier_every and req_seq % srv.inject_outlier_every == 0) else 0.0
                    time.sleep(0.022 + 0.003 * min(load, srv.max_slots) + queue_wait + outlier_pause)

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()

                    itl_sec = 0.007 + 0.002 * max(0, min(load, srv.max_slots) - 1)
                    for i in range(max_tokens):
                        if i > 0:
                            time.sleep(itl_sec)
                        chunk = {
                            "id": f"chatcmpl-gate-{req_seq}",
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": f"tok{i} "}, "finish_reason": None}],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                        self.wfile.flush()

                    final = {
                        "id": f"chatcmpl-gate-{req_seq}",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": max_tokens,
                            "total_tokens": prompt_tokens + max_tokens,
                        },
                    }
                    self.wfile.write(f"data: {json.dumps(final)}\n\n".encode("utf-8"))
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                finally:
                    with srv._lock:
                        srv._inflight -= 1

        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        t.start()

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()


def send_stream_request(port: int, req_id: str, phase: str, concurrency: int, max_tokens: int = 8):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    body = json.dumps({
        "model": "HuggingFaceTB/SmolLM-135M-Instruct",
        "messages": [{"role": "user", "content": "Benchmark Week 6 Gate streaming latency."}],
        "prompt_tokens": 12,
        "max_tokens": max_tokens,
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")

    t0 = time.perf_counter()
    ts = []
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                obj = json.loads(payload)
                if obj.get("choices") and obj["choices"][0].get("delta", {}).get("content"):
                    ts.append(time.perf_counter())
        t_end = time.perf_counter()
        rec = compute_stream_metrics(req_id, phase, t0, ts, t_end, 12, len(ts)).to_dict()
        rec["concurrency"] = concurrency
        rec["raw_trace"] = {"t_start": t0, "token_ts": ts, "t_end": t_end}
        return rec
    except urllib.error.HTTPError as e:
        t_end = time.perf_counter()
        return {
            "request_id": req_id,
            "phase": phase,
            "concurrency": concurrency,
            "status": "error",
            "ttft_ms": None,
            "mean_itl_ms": None,
            "total_latency_ms": round((t_end - t0) * 1000.0, 3),
            "output_tokens": 0,
            "output_tokens_per_sec": None,
            "error_type": f"HTTPError_{e.code}",
            "error_message": e.reason,
        }


def write_svg(rows, path):
    w, h = 760, 400
    pl, pr, pt, pb = 75, 45, 45, 60
    pw, ph = w - pl - pr, h - pt - pb
    mx = max(r["system_tps"] for r in rows) * 1.15
    my = max(r["e2e_p95_ms"] for r in rows) * 1.15
    sx = lambda v: pl + (v / mx) * pw
    sy = lambda v: pt + ph - (v / my) * ph
    p50 = " ".join(f"{sx(r['system_tps']):.1f},{sy(r['e2e_p50_ms']):.1f}" for r in rows)
    p95 = " ".join(f"{sx(r['system_tps']):.1f},{sy(r['e2e_p95_ms']):.1f}" for r in rows)
    pts = "".join(
        f'<circle cx="{sx(r["system_tps"]):.1f}" cy="{sy(r["e2e_p50_ms"]):.1f}" r="5" fill="#2563eb"/>'
        f'<circle cx="{sx(r["system_tps"]):.1f}" cy="{sy(r["e2e_p95_ms"]):.1f}" r="5" fill="#dc2626"/>'
        f'<text x="{sx(r["system_tps"])+7:.1f}" y="{sy(r["e2e_p95_ms"])-6:.1f}" font-size="11" fill="#111827">c={r["concurrency"]} ({r["system_tps"]} tok/s)</text>'
        for r in rows
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="background:#fff;font-family:sans-serif">
  <text x="{w/2}" y="25" text-anchor="middle" font-size="15" font-weight="bold">Week 6 Gate: Latency (p50/p95) vs. System Throughput</text>
  <line x1="{pl}" y1="{pt}" x2="{pl}" y2="{pt+ph}" stroke="#374151" stroke-width="1.5"/>
  <line x1="{pl}" y1="{pt+ph}" x2="{pl+pw}" y2="{pt+ph}" stroke="#374151" stroke-width="1.5"/>
  <polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{p50}"/>
  <polyline fill="none" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="5,4" points="{p95}"/>
  {pts}
  <text x="{pl+pw/2}" y="{h-16}" text-anchor="middle" font-size="13">System Throughput (tokens/sec)</text>
  <text x="18" y="{pt+ph/2}" text-anchor="middle" font-size="13" transform="rotate(-90 18,{pt+ph/2})">End-to-End Latency (ms)</text>
</svg>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


def main():
    levels = [1, 2, 4, 8]
    warmup_n = 2
    reps_n = 6
    max_tokens = 8

    # Part A: Clean Benchmark Sweep
    srv = GateBenchmarkServer(PORT, max_slots=4, max_inflight=12, inject_outlier_every=11)
    srv.start()
    all_records = []
    sweep_rows = []
    try:
        for c in levels:
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                for f in [pool.submit(send_stream_request, PORT, f"c{c}-w{i}", "warmup", c, max_tokens) for i in range(warmup_n)]:
                    all_records.append(f.result())
            w0 = time.perf_counter()
            lvl_recs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                for f in concurrent.futures.as_completed([pool.submit(send_stream_request, PORT, f"c{c}-m{i}", "measured", c, max_tokens) for i in range(reps_n)]):
                    r = f.result()
                    lvl_recs.append(r)
                    all_records.append(r)
            w_sec = time.perf_counter() - w0
            ok = [r for r in lvl_recs if r["status"] == "ok"]
            sweep_rows.append({
                "concurrency": c,
                "measured_n": reps_n,
                "ok_n": len(ok),
                "failure_rate": round((reps_n - len(ok)) / reps_n, 4),
                "system_tps": round(sum(r["output_tokens"] for r in ok) / w_sec, 2),
                "ttft_p50_ms": nearest_rank_percentile([r["ttft_ms"] for r in ok], 50),
                "ttft_p95_ms": nearest_rank_percentile([r["ttft_ms"] for r in ok], 95),
                "itl_p50_ms": nearest_rank_percentile([r["mean_itl_ms"] for r in ok], 50),
                "e2e_p50_ms": nearest_rank_percentile([r["total_latency_ms"] for r in ok], 50),
                "e2e_p95_ms": nearest_rank_percentile([r["total_latency_ms"] for r in ok], 95),
            })
    finally:
        srv.stop()

    # Part B: Independent Calculation on One Small Sample
    sample = next(r for r in all_records if r["phase"] == "measured" and r["status"] == "ok" and r["concurrency"] == 1)
    tr = sample["raw_trace"]
    ind_ttft = round((tr["token_ts"][0] - tr["t_start"]) * 1000.0, 3)
    ind_deltas = [(tr["token_ts"][i] - tr["token_ts"][i - 1]) * 1000.0 for i in range(1, len(tr["token_ts"]))]
    ind_itl = round(sum(ind_deltas) / len(ind_deltas), 3)
    ind_e2e = round((tr["t_end"] - tr["t_start"]) * 1000.0, 3)
    ind_tps = round(len(tr["token_ts"]) / (ind_e2e / 1000.0), 3)

    assert abs(ind_ttft - sample["ttft_ms"]) <= 0.002
    assert abs(ind_itl - sample["mean_itl_ms"]) <= 0.002
    assert abs(ind_e2e - sample["total_latency_ms"]) <= 0.002
    assert abs(ind_tps - sample["output_tokens_per_sec"]) <= 0.005

    # Part C: Identify Tail Outlier in Measured Set
    measured_ok = [r for r in all_records if r["phase"] == "measured" and r["status"] == "ok"]
    outlier = max(measured_ok, key=lambda r: r["total_latency_ms"])
    median_e2e = nearest_rank_percentile([r["total_latency_ms"] for r in measured_ok], 50)

    # Part D: Bounded Assumption Break (max_inflight=1 under c=4 burst) -> Observe Failure -> Restore
    broken_srv = GateBenchmarkServer(PORT + 1, max_slots=4, max_inflight=1, inject_outlier_every=0)
    broken_srv.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            broken_recs = [f.result() for f in concurrent.futures.as_completed([pool.submit(send_stream_request, PORT + 1, f"break-{i}", "measured", 4, 4) for i in range(6)])]
    finally:
        broken_srv.stop()
    broken_fails = [r for r in broken_recs if r["status"] != "ok"]

    restored_srv = GateBenchmarkServer(PORT + 2, max_slots=4, max_inflight=12, inject_outlier_every=0)
    restored_srv.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            restored_recs = [f.result() for f in concurrent.futures.as_completed([pool.submit(send_stream_request, PORT + 2, f"restore-{i}", "measured", 4, 4) for i in range(6)])]
    finally:
        restored_srv.stop()
    restored_fails = [r for r in restored_recs if r["status"] != "ok"]

    os.makedirs("artifacts/day41", exist_ok=True)
    with open("artifacts/day41/week06-sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(sweep_rows)

    write_svg(sweep_rows, "artifacts/day41/week06-latency-throughput.svg")

    clean_records = [{k: v for k, v in r.items() if k != "raw_trace"} for r in all_records]
    overall_summary = summarize_records(clean_records, source_file="week06-gate-in-memory")

    gate_bundle = {
        "gate": "Week 6 Benchmark Gate (Day 41)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.getoutput("git rev-parse --short HEAD").strip(),
        "environment": {"os": platform.platform(), "python": sys.version.split()[0], "cpu_cores": os.cpu_count()},
        "reproduce_command": "./.venv-linux/bin/python scripts/run_week06_gate.py",
        "sweep_results": sweep_rows,
        "overall_summary": overall_summary,
        "independent_sample_verification": {
            "request_id": sample["request_id"],
            "harness_ttft_ms": sample["ttft_ms"],
            "independent_ttft_ms": ind_ttft,
            "harness_mean_itl_ms": sample["mean_itl_ms"],
            "independent_mean_itl_ms": ind_itl,
            "harness_e2e_ms": sample["total_latency_ms"],
            "independent_e2e_ms": ind_e2e,
            "harness_tps": sample["output_tokens_per_sec"],
            "independent_tps": ind_tps,
            "verified_match": True,
        },
        "identified_outlier": {
            "request_id": outlier["request_id"],
            "concurrency": outlier["concurrency"],
            "total_latency_ms": outlier["total_latency_ms"],
            "ttft_ms": outlier["ttft_ms"],
            "global_e2e_p50_ms": median_e2e,
            "slowdown_vs_p50_ratio": round(outlier["total_latency_ms"] / median_e2e, 2),
        },
        "bounded_assumption_break_test": {
            "broken_assumption": "Set admission limit max_inflight=1 under concurrency=4 burst",
            "broken_failure_rate": round(len(broken_fails) / len(broken_recs), 4),
            "observed_symptom": broken_fails[0]["error_type"] if broken_fails else "none",
            "restored_assumption": "Restored max_inflight=12",
            "restored_failure_rate": round(len(restored_fails) / len(restored_recs), 4),
        },
    }
    with open("artifacts/day41/week06-benchmark.json", "w", encoding="utf-8") as f:
        json.dump(gate_bundle, f, indent=2)

    print("=== WEEK 6 GATE SWEEP ===")
    for r in sweep_rows:
        print(r)
    print("\n=== INDEPENDENT SAMPLE CHECK ===")
    print(json.dumps(gate_bundle["independent_sample_verification"], indent=2))
    print("\n=== IDENTIFIED OUTLIER ===")
    print(json.dumps(gate_bundle["identified_outlier"], indent=2))
    print("\n=== BOUNDED ASSUMPTION BREAK & RESTORE ===")
    print(json.dumps(gate_bundle["bounded_assumption_break_test"], indent=2))


if __name__ == "__main__":
    main()
