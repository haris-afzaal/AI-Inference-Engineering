#!/usr/bin/env python3
"""
Day 39: Concurrency Sweep (1, 2, 4, 8, 12) with 5 measured repetitions per level.
Models continuous-batching slot contention, queueing delay, ITL degradation, and
admission saturation (HTTP 503), emitting CSV, SVG/PNG plot, and manifest.
"""
import concurrent.futures
import csv
import json
import os
import platform
import statistics
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

PORT = 8039
MAX_ACTIVE_SLOTS = 4   # Parallel KV cache batch slots
MAX_TOTAL_INFLIGHT = 8 # Active + waiting queue admission limit


class CapacityBoundedEngine:
    def __init__(self, port: int):
        self.port = port
        self._lock = threading.Lock()
        self._inflight = 0
        self.httpd = None

    def start(self):
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_POST(self):
                if self.path != "/v1/chat/completions":
                    self.send_response(404)
                    self.end_headers()
                    return

                with engine._lock:
                    if engine._inflight >= MAX_TOTAL_INFLIGHT:
                        err = json.dumps({
                            "error": {
                                "message": f"Queue saturated: inflight={engine._inflight} >= limit={MAX_TOTAL_INFLIGHT}",
                                "type": "server_overload",
                                "code": "queue_full_503",
                            }
                        }).encode("utf-8")
                        self.send_response(503)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(err)))
                        self.end_headers()
                        self.wfile.write(err)
                        return
                    engine._inflight += 1
                    current_load = engine._inflight

                try:
                    content_len = int(self.headers.get("Content-Length", 0))
                    payload = json.loads(self.rfile.read(content_len).decode("utf-8"))
                    max_tokens = int(payload.get("max_tokens", 16))
                    prompt_tokens = 12

                    # Signal 1: Queueing + Prefill TTFT scales sharply once load > MAX_ACTIVE_SLOTS
                    queue_excess = max(0, current_load - MAX_ACTIVE_SLOTS)
                    prefill_delay_sec = 0.025 + (0.004 * min(current_load, MAX_ACTIVE_SLOTS)) + (0.045 * queue_excess)
                    time.sleep(prefill_delay_sec)

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()

                    # Signal 2: Memory-bandwidth contention degrades ITL as active batch grows
                    with engine._lock:
                        active_batch = min(engine._inflight, MAX_ACTIVE_SLOTS)
                    itl_delay_sec = 0.008 + 0.0025 * (active_batch - 1) + (0.006 * queue_excess)

                    for i in range(max_tokens):
                        if i > 0:
                            time.sleep(itl_delay_sec)
                        chunk = {
                            "id": "chatcmpl-day39",
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": f"w{i} "}, "finish_reason": None}],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                        self.wfile.flush()

                    final_chunk = {
                        "id": "chatcmpl-day39",
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
                finally:
                    with engine._lock:
                        engine._inflight -= 1

        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        t.start()

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()


def single_call(req_id: str, phase: str, prompt: str, max_tokens: int):
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    body = json.dumps({
        "model": "HuggingFaceTB/SmolLM-135M-Instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")

    t0 = time.perf_counter()
    timestamps = []
    usage = None
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
                    timestamps.append(time.perf_counter())
                if obj.get("usage"):
                    usage = obj["usage"]
        t_end = time.perf_counter()
        rec = compute_stream_metrics(
            request_id=req_id,
            phase=phase,
            t_start_sec=t0,
            token_timestamps_sec=timestamps,
            t_end_sec=t_end,
            prompt_tokens=usage["prompt_tokens"] if usage else 12,
            output_tokens=usage["completion_tokens"] if usage else len(timestamps),
        )
        return rec.to_dict()
    except urllib.error.HTTPError as e:
        t_end = time.perf_counter()
        return {
            "request_id": req_id,
            "phase": phase,
            "status": "error",
            "http_status": e.code,
            "ttft_ms": None,
            "mean_itl_ms": None,
            "total_latency_ms": round((t_end - t0) * 1000.0, 3),
            "output_tokens": 0,
            "error_type": f"HTTPError_{e.code}",
        }


def run_level(concurrency: int, warmup: int, repetitions: int, prompt: str, max_tokens: int):
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(single_call, f"c{concurrency}-w{i}", "warmup", prompt, max_tokens) for i in range(warmup)]
        for f in futs:
            f.result()

    wall_start = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futs = [pool.submit(single_call, f"c{concurrency}-m{i}", "measured", prompt, max_tokens) for i in range(repetitions)]
        for f in concurrent.futures.as_completed(futs):
            results.append(f.result())
    wall_sec = time.perf_counter() - wall_start

    ok = [r for r in results if r["status"] == "ok"]
    err = [r for r in results if r["status"] != "ok"]
    fail_rate = round(len(err) / repetitions, 4)
    total_out_tokens = sum(r["output_tokens"] for r in ok)
    sys_tps = round(total_out_tokens / wall_sec, 2) if wall_sec > 0 else 0.0

    ttfts = [r["ttft_ms"] for r in ok]
    itls = [r["mean_itl_ms"] for r in ok]
    lats = [r["total_latency_ms"] for r in ok]

    return {
        "concurrency": concurrency,
        "measured_n": repetitions,
        "ok_n": len(ok),
        "error_n": len(err),
        "failure_rate": fail_rate,
        "system_throughput_tps": sys_tps,
        "ttft_p50_ms": nearest_rank_percentile(ttfts, 50),
        "ttft_p95_ms": nearest_rank_percentile(ttfts, 95),
        "itl_p50_ms": nearest_rank_percentile(itls, 50),
        "itl_p95_ms": nearest_rank_percentile(itls, 95),
        "e2e_p50_ms": nearest_rank_percentile(lats, 50),
        "e2e_p95_ms": nearest_rank_percentile(lats, 95),
    }


def generate_svg_plot(rows, svg_path):
    width, height = 760, 420
    pad_l, pad_r, pad_t, pad_b = 75, 40, 45, 65
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    valid = [r for r in rows if r["e2e_p50_ms"] is not None]
    max_x = max(r["system_throughput_tps"] for r in valid) * 1.15
    max_y = max(r["e2e_p95_ms"] for r in valid) * 1.15

    def sx(val):
        return pad_l + (val / max_x) * plot_w

    def sy(val):
        return pad_t + plot_h - (val / max_y) * plot_h

    p50_pts = " ".join(f"{sx(r['system_throughput_tps']):.1f},{sy(r['e2e_p50_ms']):.1f}" for r in valid)
    p95_pts = " ".join(f"{sx(r['system_throughput_tps']):.1f},{sy(r['e2e_p95_ms']):.1f}" for r in valid)

    circles = []
    for r in valid:
        cx, cy50, cy95 = sx(r["system_throughput_tps"]), sy(r["e2e_p50_ms"]), sy(r["e2e_p95_ms"])
        lbl = f"c={r['concurrency']} ({r['system_throughput_tps']} tok/s, fail={int(r['failure_rate']*100)}%)"
        circles.append(f'<circle cx="{cx:.1f}" cy="{cy50:.1f}" r="5" fill="#2563eb"/>')
        circles.append(f'<circle cx="{cx:.1f}" cy="{cy95:.1f}" r="5" fill="#dc2626"/>')
        circles.append(f'<text x="{cx + 7:.1f}" y="{cy95 - 6:.1f}" font-size="11" fill="#1f2937">{lbl}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background:#ffffff;font-family:sans-serif">
  <text x="{width/2}" y="26" text-anchor="middle" font-size="15" font-weight="bold" fill="#111827">Day 39 Concurrency Sweep: Latency vs. System Throughput</text>
  <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t+plot_h}" stroke="#374151" stroke-width="1.5"/>
  <line x1="{pad_l}" y1="{pad_t+plot_h}" x2="{pad_l+plot_w}" y2="{pad_t+plot_h}" stroke="#374151" stroke-width="1.5"/>
  <polyline fill="none" stroke="#2563eb" stroke-width="2.5" points="{p50_pts}"/>
  <polyline fill="none" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="5,4" points="{p95_pts}"/>
  {''.join(circles)}
  <text x="{pad_l + plot_w/2}" y="{height - 18}" text-anchor="middle" font-size="13" fill="#111827">Aggregate System Throughput (output tokens/sec)</text>
  <text x="20" y="{pad_t + plot_h/2}" text-anchor="middle" font-size="13" fill="#111827" transform="rotate(-90 20,{pad_t + plot_h/2})">End-to-End Latency (ms)</text>
  <rect x="{width-220}" y="{pad_t+8}" width="185" height="48" fill="#f9fafb" stroke="#d1d5db"/>
  <line x1="{width-210}" y1="{pad_t+24}" x2="{width-185}" y2="{pad_t+24}" stroke="#2563eb" stroke-width="2.5"/>
  <text x="{width-178}" y="{pad_t+28}" font-size="11" fill="#111827">p50 E2E Latency (ms)</text>
  <line x1="{width-210}" y1="{pad_t+42}" x2="{width-185}" y2="{pad_t+42}" stroke="#dc2626" stroke-width="2.5" stroke-dasharray="5,4"/>
  <text x="{width-178}" y="{pad_t+46}" font-size="11" fill="#111827">p95 E2E Latency (ms)</text>
</svg>"""
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)


def main():
    levels = [1, 2, 4, 8, 12]
    warmup = 2
    repetitions = 5
    max_tokens = 16
    prompt = "Explain how continuous batching improves GPU utilization under concurrent load."

    engine = CapacityBoundedEngine(PORT)
    engine.start()
    rows = []
    try:
        for c in levels:
            row = run_level(c, warmup=warmup, repetitions=repetitions, prompt=prompt, max_tokens=max_tokens)
            rows.append(row)
    finally:
        engine.stop()

    os.makedirs("artifacts/day39", exist_ok=True)
    csv_path = "artifacts/day39/day39-concurrency.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    svg_path = "artifacts/day39/latency_vs_throughput.svg"
    generate_svg_plot(rows, svg_path)

    manifest = {
        "day": 39,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "os": platform.platform(),
            "python_version": sys.version.split()[0],
            "cpu_cores": os.cpu_count(),
        },
        "workload_contract": {
            "model": "HuggingFaceTB/SmolLM-135M-Instruct",
            "max_tokens": max_tokens,
            "warmup_per_level": warmup,
            "measured_repetitions_per_level": repetitions,
            "concurrency_levels": levels,
            "engine_slots": MAX_ACTIVE_SLOTS,
            "engine_queue_limit": MAX_TOTAL_INFLIGHT,
        },
        "results": rows,
    }
    with open("artifacts/day39/manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=== DAY 39 CONCURRENCY SWEEP TABLE ===")
    header = f"{'Conc':>4} | {'OK/N':>5} | {'Fail%':>6} | {'Sys TPS':>8} | {'TTFT p50':>9} | {'TTFT p95':>9} | {'ITL p50':>8} | {'E2E p50':>8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['concurrency']:>4} | {r['ok_n']}/{r['measured_n']:>2} | {r['failure_rate']*100:>5.1f}% | "
            f"{r['system_throughput_tps']:>8.2f} | {str(r['ttft_p50_ms']):>9} | {str(r['ttft_p95_ms']):>9} | "
            f"{str(r['itl_p50_ms']):>8} | {str(r['e2e_p50_ms']):>8}"
        )


if __name__ == "__main__":
    main()
