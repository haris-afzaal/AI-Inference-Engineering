import json
import time

def run_vllm_minilab():
    print("=== vLLM Configuration Mini-Lab ===\n")
    
    # Mock parameters and their expected real-world effects
    configs = [
        {"name": "Baseline (Safe)", "gpu_mem": 0.8, "max_len": 2048, "ttft_ms": 45, "throughput": 1500, "gpu_blocks": 3500},
        {"name": "High Concurrency", "gpu_mem": 0.95, "max_len": 2048, "ttft_ms": 52, "throughput": 2800, "gpu_blocks": 8500},
        {"name": "Long Context (16k)", "gpu_mem": 0.9, "max_len": 16384, "ttft_ms": 120, "throughput": 800, "gpu_blocks": 1500} 
    ]
    
    results = []
    
    for c in configs:
        print(f"--- Starting vLLM Engine: [{c['name']}] ---")
        
        # Simulating vLLM container startup logs
        print(f"INFO 11-13 11:30:00 engine.py] Initializing an LLM engine with config: gpu_memory_utilization={c['gpu_mem']}, max_model_len={c['max_len']}")
        time.sleep(0.3)
        print("INFO 11-13 11:30:01 worker.py] Profiling run to measure memory usage. Please wait...")
        time.sleep(0.5)
        
        # The most important vLLM log line: How many blocks it actually allocated
        print(f"INFO 11-13 11:30:03 cache_engine.py] # GPU blocks: {c['gpu_blocks']}, # CPU blocks: 512")
        print("INFO 11-13 11:30:04 server.py] Application startup complete.")
        
        # Simulating the Benchmark Workload
        print(f"\n[Benchmarking Workload - {c['name']}]")
        print(f" -> Latency (TTFT) : {c['ttft_ms']} ms")
        print(f" -> Throughput     : {c['throughput']} tokens/sec")
        print("-" * 50 + "\n")
        
        results.append(c)
        
    with open("artifacts/day34/vllm_minilab_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Logs saved to artifacts/day34/vllm_minilab_results.json")

if __name__ == "__main__":
    run_vllm_minilab()
