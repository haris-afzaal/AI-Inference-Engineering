#!/usr/bin/env python3
"""
Day 37: Instrument SSE stream chunks to record per-token deltas, TTFT, mean ITL,
total latency, throughput, and run manifest metadata.
"""
import json
import os
import platform
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.abspath("src"))
from inference_client.metrics import compute_stream_metrics, summarize_measured_records

PORT = 8037


class TimedSSEHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        content_len = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(content_len).decode("utf-8"))
        prompt_text = payload["messages"][0]["content"]
        prompt_tokens = max(5, len(prompt_text.split()))

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        # Simulate prefill latency
        time.sleep(0.035)
        tokens = ["TTFT ", "measures ", "prefill, ", "ITL ", "measures ", "decode."]

        for i, tok in enumerate(tokens):
            if i > 0:
                time.sleep(0.012 + (i % 2) * 0.004)
            chunk = {
                "id": "chatcmpl-day37",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}],
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()

        final_chunk = {
            "id": "chatcmpl-day37",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(tokens),
                "total_tokens": prompt_tokens + len(tokens),
            },
        }
        self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def measure_stream_request(req_id, phase, prompt_text):
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    body = json.dumps({
        "model": "HuggingFaceTB/SmolLM-135M-Instruct",
        "messages": [{"role": "user", "content": prompt_text}],
        "stream": True,
    }).encode("utf-8")

    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    t_start = time.perf_counter()
    token_timestamps = []
    usage = None

    with urllib.request.urlopen(req, timeout=5) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            parsed = json.loads(data_str)
            choices = parsed.get("choices", [])
            if choices and choices[0].get("delta", {}).get("content"):
                token_timestamps.append(time.perf_counter())
            if parsed.get("usage"):
                usage = parsed["usage"]

    t_end = time.perf_counter()
    prompt_tokens = usage["prompt_tokens"] if usage else len(prompt_text.split())
    output_tokens = usage["completion_tokens"] if usage else len(token_timestamps)

    return compute_stream_metrics(
        request_id=req_id,
        phase=phase,
        t_start_sec=t_start,
        token_timestamps_sec=token_timestamps,
        t_end_sec=t_end,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
    )


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), TimedSSEHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    prompts = [
        "Explain TTFT vs ITL in LLM streaming.",
        "How does prefill differ from autoregressive decode?",
        "Why do per-token deltas vary under continuous batching?",
        "Compare end-to-end TPS with decode-only TPS.",
        "What causes tail spikes in inter-token latency?",
    ]

    records = []
    try:
        # 2 warm-up + 5 measured requests
        for i in range(2):
            records.append(measure_stream_request(f"warmup-{i}", "warmup", prompts[i % len(prompts)]))
        for i in range(5):
            records.append(measure_stream_request(f"measured-{i}", "measured", prompts[i % len(prompts)]))
    finally:
        srv.shutdown()

    os.makedirs("artifacts/day37", exist_ok=True)
    jsonl_path = "artifacts/day37/day37-latency.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    summary = summarize_measured_records(records)
    manifest = {
        "day": 37,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "os": platform.platform(),
            "python_version": sys.version.split()[0],
            "cpu_cores": os.cpu_count(),
        },
        "workload": {
            "model": "HuggingFaceTB/SmolLM-135M-Instruct",
            "warmup_count": 2,
            "repetitions": 5,
            "concurrency": 1,
        },
        "summary": summary,
    }
    manifest_path = "artifacts/day37/manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("=== DAY 37 MANIFEST ===")
    print(json.dumps(manifest, indent=2))
    print("\n=== SAMPLE MEASURED RECORD ===")
    print(json.dumps(records[-1].to_dict(), indent=2))


if __name__ == "__main__":
    main()
