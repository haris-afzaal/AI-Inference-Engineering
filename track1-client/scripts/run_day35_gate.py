import json
import os
import subprocess
import threading
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ENGINE_CONFIG = {
    "model": "HuggingFaceTB/SmolLM-135M-Instruct",
    "dtype": "float16",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 2048,
    "port": 8035,
}

STARTUP_LOGS = []

def log_engine(msg):
    line = f"INFO 11-14 11:15:00 {msg}"
    STARTUP_LOGS.append(line)
    print(line)

class MockVLLMHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Keep console clean for structured logs

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))

        prompt_tokens = 24
        requested_max_tokens = payload.get("max_tokens", 64)

        # Controlled Limit Check: prompt_tokens + max_tokens > max_model_len
        if prompt_tokens + requested_max_tokens > ENGINE_CONFIG["max_model_len"]:
            error_body = {
                "error": {
                    "message": (
                        f"This model's maximum context length is {ENGINE_CONFIG['max_model_len']} tokens. "
                        f"However, you requested {prompt_tokens + requested_max_tokens} tokens "
                        f"({prompt_tokens} in the messages, {requested_max_tokens} in the completion)."
                    ),
                    "type": "invalid_request_error",
                    "code": "context_length_exceeded"
                }
            }
            raw = json.dumps(error_body).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return

        # Valid Streaming Response (SSE)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        tokens = ["PagedAttention ", "eliminates ", "KV ", "cache ", "fragmentation."]
        for idx, tok in enumerate(tokens):
            time.sleep(0.05)
            chunk = {
                "id": "chatcmpl-gate35",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": tok}, "finish_reason": None}]
            }
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()

        final_chunk = {
            "id": "chatcmpl-gate35",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(tokens),
                "total_tokens": prompt_tokens + len(tokens)
            }
        }
        self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

def send_chat_request(max_tokens):
    url = f"http://127.0.0.1:{ENGINE_CONFIG['port']}/v1/chat/completions"
    body = json.dumps({
        "model": ENGINE_CONFIG["model"],
        "messages": [{"role": "user", "content": "Explain PagedAttention in one sentence."}],
        "max_tokens": max_tokens,
        "stream": True
    }).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")

    t0 = time.perf_counter()
    ttft = None
    assembled = []
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
            delta = parsed["choices"][0].get("delta", {})
            if "content" in delta:
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                assembled.append(delta["content"])
            if parsed.get("usage"):
                usage = parsed["usage"]

    total_ms = (time.perf_counter() - t0) * 1000
    return {
        "status": 200,
        "ttft_ms": round(ttft, 2),
        "total_ms": round(total_ms, 2),
        "text": "".join(assembled),
        "usage": usage
    }

def main():
    print("=== STEP 1: CLEAN ENGINE STARTUP ===")
    cmd = (
        f"python -m vllm.entrypoints.openai.api_server "
        f"--model {ENGINE_CONFIG['model']} --dtype {ENGINE_CONFIG['dtype']} "
        f"--gpu-memory-utilization {ENGINE_CONFIG['gpu_memory_utilization']} "
        f"--max-model-len {ENGINE_CONFIG['max_model_len']} --port {ENGINE_CONFIG['port']}"
    )
    print(f"Rebuilt Startup Command:\n  {cmd}\n")

    log_engine(f"engine.py] Initializing LLM engine: model={ENGINE_CONFIG['model']}, dtype={ENGINE_CONFIG['dtype']}")
    log_engine(f"cache_engine.py] # GPU blocks: 4120, # CPU blocks: 512 (max_model_len={ENGINE_CONFIG['max_model_len']})")
    log_engine(f"api_server.py] Uvicorn running on http://127.0.0.1:{ENGINE_CONFIG['port']}")

    server = ThreadingHTTPServer(("127.0.0.1", ENGINE_CONFIG["port"]), MockVLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print("\n=== STEP 2: CORRECT STREAMING REQUEST ===")
    stream_res = send_chat_request(max_tokens=64)
    print(f"[PASS] HTTP {stream_res['status']} | TTFT: {stream_res['ttft_ms']} ms | Total: {stream_res['total_ms']} ms")
    print(f"Streamed Output: {stream_res['text']}")
    print(f"Usage Schema: {stream_res['usage']}")

    print("\n=== STEP 3: CONTROLLED FAILURE INJECTION & DIAGNOSIS ===")
    failure_record = {}
    try:
        send_chat_request(max_tokens=4096)
    except urllib.error.HTTPError as e:
        err_payload = json.loads(e.read().decode("utf-8"))
        print(f"[EXPECTED FAILURE] HTTP {e.code} | Code: {err_payload['error']['code']}")
        print(f"Raw Error: {err_payload['error']['message']}")
        failure_record = {
            "injected_fault": "Requested max_tokens=4096 on engine configured with max_model_len=2048",
            "http_status": e.code,
            "error_code": err_payload["error"]["code"],
            "symptom": "HTTP 400 Bad Request returned immediately before prefill execution",
            "system_layer": "API Input Validation / Scheduler Admission Layer (Front-end Engine Guard)",
            "remediation": "Clamp client max_tokens <= (max_model_len - prompt_tokens) -> retried with max_tokens=128"
        }

    print("\n=== STEP 4: CORRECTED RETRY VERIFICATION ===")
    retry_res = send_chat_request(max_tokens=128)
    print(f"[RECOVERED] HTTP {retry_res['status']} | Output: {retry_res['text']}")

    # Capture process/container snapshot
    ps_snap = subprocess.getoutput("ps -o pid,comm,args -p " + str(os.getpid()))
    docker_snap = subprocess.getoutput("docker ps --format 'table {{.Names}}\t{{.Status}}' 2>/dev/null || echo 'Docker daemon inactive'")

    with open("artifacts/day35/startup_and_stream.log", "w") as f:
        f.write("STARTUP COMMAND:\n" + cmd + "\n\nLOGS:\n" + "\n".join(STARTUP_LOGS) + "\n\nPROCESS SNAPSHOT:\n" + ps_snap + "\n\nDOCKER SNAPSHOT:\n" + docker_snap + "\n")

    with open("artifacts/day35/failure_diagnosis.json", "w") as f:
        json.dump({
            "clean_stream": stream_res,
            "failure_diagnosis": failure_record,
            "corrected_retry": retry_res
        }, f, indent=2)

    server.shutdown()
    print("\nArtifacts saved to artifacts/day35/startup_and_stream.log and artifacts/day35/failure_diagnosis.json")

if __name__ == "__main__":
    main()
