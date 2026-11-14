#!/usr/bin/env python3
"""
Day 36: Reproducible LLM Inference Benchmark Harness CLI.
Supports OpenAI-compatible SSE endpoints, warm-up isolation, bounded concurrency,
fault-tolerant JSONL recording, schema validation, and run manifest metadata.
"""
import argparse
import concurrent.futures
import json
import os
import platform
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def load_prompts(prompt_file):
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                obj = json.loads(line)
                prompts.append({
                    "prompt_id": obj.get("id", f"p-{idx}"),
                    "text": obj["prompt"]
                })
            else:
                prompts.append({"prompt_id": f"p-{idx}", "text": line})
    if not prompts:
        raise ValueError(f"No valid prompts found in {prompt_file}")
    return prompts


def validate_record_against_schema(record, schema):
    for req_key in schema["required"]:
        if req_key not in record:
            raise AssertionError(f"Missing required field: {req_key}")
    for key in record:
        if key not in schema["properties"]:
            raise AssertionError(f"Unexpected extra field: {key}")
    if record["phase"] not in ("warmup", "measured"):
        raise AssertionError(f"Invalid phase: {record['phase']}")
    if record["status"] not in ("ok", "error"):
        raise AssertionError(f"Invalid status: {record['status']}")
    if record["status"] == "ok":
        assert record["ttft_ms"] is not None and record["total_latency_ms"] >= record["ttft_ms"]
        assert record["error_type"] is None
    else:
        assert record["error_type"] is not None


class BuiltinMockInferenceServer:
    """Deterministic local SSE server with configurable fault injection for harness verification."""
    def __init__(self, port, fail_every=0):
        self.port = port
        self.fail_every = fail_every
        self._counter = 0
        self._lock = threading.Lock()
        self.httpd = None

    def start(self):
        parent = self

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
                max_tokens = min(int(payload.get("max_tokens", 16)), 16)

                with parent._lock:
                    parent._counter += 1
                    req_num = parent._counter

                if parent.fail_every > 0 and (req_num % parent.fail_every == 0):
                    err = json.dumps({
                        "error": {
                            "message": "Simulated engine queue overload (503 Service Unavailable)",
                            "type": "server_overload",
                            "code": "engine_overloaded"
                        }
                    }).encode("utf-8")
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                # Simulate prefill delay
                time.sleep(0.025)
                prompt_text = payload["messages"][0]["content"]
                prompt_tokens = max(4, len(prompt_text.split()))

                for i in range(max_tokens):
                    if i > 0:
                        time.sleep(0.008)
                    chunk = {
                        "id": f"chatcmpl-bench-{req_num}",
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": f"tok{i} "}, "finish_reason": None}]
                    }
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                    self.wfile.flush()

                final_chunk = {
                    "id": f"chatcmpl-bench-{req_num}",
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": max_tokens,
                        "total_tokens": prompt_tokens + max_tokens
                    }
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


def execute_single_request(run_id, phase, req_idx, prompt_item, args):
    url = args.model_url.rstrip("/") + "/v1/chat/completions"
    req_seed = args.seed + req_idx
    body = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": prompt_item["text"]}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "seed": req_seed,
        "stream": True,
        "stream_options": {"include_usage": True}
    }).encode("utf-8")

    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    ts_utc = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    ttft_ms = None
    chunks_seen = 0
    usage = None

    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            http_status = resp.status
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload_str = line[6:]
                if payload_str == "[DONE]":
                    break
                parsed = json.loads(payload_str)
                choices = parsed.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if delta.get("content"):
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000.0
                        chunks_seen += 1
                if parsed.get("usage"):
                    usage = parsed["usage"]

        total_ms = (time.perf_counter() - t0) * 1000.0
        comp_tokens = usage["completion_tokens"] if usage else chunks_seen
        prompt_tokens = usage["prompt_tokens"] if usage else len(prompt_item["text"].split())

        if ttft_ms is None:
            ttft_ms = total_ms
        decode_ms = max(0.0, total_ms - ttft_ms)
        tpot_ms = (decode_ms / max(1, comp_tokens - 1)) if comp_tokens > 1 else 0.0
        tps = (comp_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0

        return {
            "run_id": run_id,
            "phase": phase,
            "request_index": req_idx,
            "prompt_id": prompt_item["prompt_id"],
            "concurrency": args.concurrency,
            "seed": req_seed,
            "status": "ok",
            "http_status": http_status,
            "ttft_ms": round(ttft_ms, 3),
            "tpot_ms": round(tpot_ms, 3),
            "total_latency_ms": round(total_ms, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": comp_tokens,
            "tokens_per_sec": round(tps, 2),
            "error_type": None,
            "error_message": None,
            "timestamp_utc": ts_utc
        }
    except urllib.error.HTTPError as e:
        total_ms = (time.perf_counter() - t0) * 1000.0
        err_body = e.read().decode("utf-8", errors="replace")
        return {
            "run_id": run_id,
            "phase": phase,
            "request_index": req_idx,
            "prompt_id": prompt_item["prompt_id"],
            "concurrency": args.concurrency,
            "seed": req_seed,
            "status": "error",
            "http_status": e.code,
            "ttft_ms": None,
            "tpot_ms": None,
            "total_latency_ms": round(total_ms, 3),
            "prompt_tokens": None,
            "completion_tokens": None,
            "tokens_per_sec": None,
            "error_type": f"HTTPError_{e.code}",
            "error_message": err_body[:200],
            "timestamp_utc": ts_utc
        }
    except Exception as e:
        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "run_id": run_id,
            "phase": phase,
            "request_index": req_idx,
            "prompt_id": prompt_item["prompt_id"],
            "concurrency": args.concurrency,
            "seed": req_seed,
            "status": "error",
            "http_status": None,
            "ttft_ms": None,
            "tpot_ms": None,
            "total_latency_ms": round(total_ms, 3),
            "prompt_tokens": None,
            "completion_tokens": None,
            "tokens_per_sec": None,
            "error_type": type(e).__name__,
            "error_message": str(e)[:200],
            "timestamp_utc": ts_utc
        }


def run_phase(run_id, phase, count, prompts, args, schema):
    rng = random.Random(args.seed + (0 if phase == "warmup" else 10000))
    tasks = [rng.choice(prompts) for _ in range(count)]
    records = [None] * count

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        future_map = {
            pool.submit(execute_single_request, run_id, phase, idx, prompt_item, args): idx
            for idx, prompt_item in enumerate(tasks)
        }
        for fut in concurrent.futures.as_completed(future_map):
            idx = future_map[fut]
            rec = fut.result()
            validate_record_against_schema(rec, schema)
            records[idx] = rec
    return records


def collect_hardware_metadata():
    cpu_model = platform.processor() or "unknown"
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if "model name" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    gpu_info = subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null")
    if not gpu_info or "not found" in gpu_info or "failed" in gpu_info:
        gpu_info = "none (CPU / mock validation mode)"
    return {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_model": cpu_model,
        "cpu_cores": os.cpu_count(),
        "gpu": gpu_info
    }


def main():
    parser = argparse.ArgumentParser(description="Controlled LLM Inference Benchmark Harness")
    parser.add_argument("--model-url", default="http://127.0.0.1:8036", help="Base URL of OpenAI-compatible server")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM-135M-Instruct", help="Target model identifier")
    parser.add_argument("--prompt-file", required=True, help="Path to prompts (.txt or .jsonl)")
    parser.add_argument("--max-tokens", type=int, default=16, help="Max completion tokens per request")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent workers")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warm-up requests excluded from summary")
    parser.add_argument("--repetitions", type=int, default=5, help="Number of measured requests")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-request timeout in seconds")
    parser.add_argument("--output-dir", default="artifacts/day36", help="Directory for JSONL records and manifest")
    parser.add_argument("--schema", default="schemas/benchmark_record.json", help="Path to JSON schema")
    parser.add_argument("--mock-server", action="store_true", help="Launch built-in local SSE mock server")
    parser.add_argument("--inject-fail-every", type=int, default=0, help="Inject HTTP 503 every Nth request in mock mode")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)

    prompts = load_prompts(args.prompt_file)
    run_id = f"bench-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"

    mock_srv = None
    if args.mock_server:
        port = int(args.model_url.rsplit(":", 1)[-1])
        mock_srv = BuiltinMockInferenceServer(port=port, fail_every=args.inject_fail_every)
        mock_srv.start()

    try:
        print(f"[Manifest] Run ID: {run_id} | Model: {args.model} | Concurrency: {args.concurrency} | Seed: {args.seed}")
        print(f"[Phase 1/2] Executing {args.warmup} warm-up requests (excluded from measured metrics)...")
        warmup_records = run_phase(run_id, "warmup", args.warmup, prompts, args, schema)

        print(f"[Phase 2/2] Executing {args.repetitions} measured requests...")
        measured_records = run_phase(run_id, "measured", args.repetitions, prompts, args, schema)
    finally:
        if mock_srv:
            mock_srv.stop()

    all_records = warmup_records + measured_records
    records_path = os.path.join(args.output_dir, "benchmark_records.jsonl")
    with open(records_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    measured_ok = [r for r in measured_records if r["status"] == "ok"]
    measured_err = [r for r in measured_records if r["status"] == "error"]
    failure_rate = round(len(measured_err) / max(1, len(measured_records)), 4)

    ttfts = sorted(r["ttft_ms"] for r in measured_ok)
    latencies = sorted(r["total_latency_ms"] for r in measured_ok)
    p50_ttft = ttfts[len(ttfts) // 2] if ttfts else None
    p50_lat = latencies[len(latencies) // 2] if latencies else None

    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment": collect_hardware_metadata(),
        "workload_config": {
            "model_url": args.model_url,
            "model": args.model,
            "prompt_file": args.prompt_file,
            "prompt_pool_size": len(prompts),
            "max_tokens": args.max_tokens,
            "concurrency": args.concurrency,
            "warmup_count": args.warmup,
            "measured_count": args.repetitions,
            "seed": args.seed
        },
        "statistical_hygiene": {
            "warmup_isolated": True,
            "schema_validated": True,
            "measured_ok_count": len(measured_ok),
            "measured_error_count": len(measured_err),
            "failure_rate": failure_rate,
            "p50_ttft_ms": p50_ttft,
            "p50_total_latency_ms": p50_lat
        }
    }

    manifest_path = os.path.join(args.output_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Completed] Records written: {len(all_records)} ({len(warmup_records)} warmup + {len(measured_records)} measured)")
    print(f"[Hygiene] Measured OK: {len(measured_ok)} | Errors Recorded: {len(measured_err)} | Failure Rate: {failure_rate * 100:.1f}%")
    print(f"[Artifacts] {records_path} & {manifest_path}")


if __name__ == "__main__":
    main()
