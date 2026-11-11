import json

def simulate_vllm_startup():
    print("--- vLLM Engine Startup Configuration Matrix ---\n")
    
    # Base simulated environment
    total_gpu_vram_gb = 8.0
    pytorch_overhead_gb = 0.5
    
    # Model: SmolLM-135M Base details
    num_params = 135_000_000
    
    # We will test a matrix of different configurations
    configs_to_test = [
        {"name": "Standard Setup", "utilization": 0.9, "max_len": 2048, "dtype": "float16"},
        {"name": "Low Mem Budget", "utilization": 0.4, "max_len": 2048, "dtype": "float16"},
        {"name": "Huge Context",   "utilization": 0.9, "max_len": 32000, "dtype": "float16"},
        {"name": "FP32 Datatype",  "utilization": 0.9, "max_len": 2048, "dtype": "float32"},
        {"name": "Over Allocating", "utilization": 1.1, "max_len": 2048, "dtype": "float16"}
    ]
    
    results = []
    
    for config in configs_to_test:
        print(f"Testing Config: [{config['name']}]")
        print(f"  Params: gpu_util={config['utilization']}, max_len={config['max_len']}, dtype={config['dtype']}")
        
        # 1. Determine bytes per parameter based on dtype
        bytes_per_param = 4 if config["dtype"] == "float32" else 2
        
        # 2. Calculate Model Weight Size
        model_weights_gb = (num_params * bytes_per_param) / (1024**3)
        
        # 3. Validation: Check if utilization is legal
        if config["utilization"] > 1.0:
            print("  Result: [FAILED] Crash! gpu_memory_utilization cannot exceed 1.0.\n")
            results.append({"config": config, "status": "FAILED", "reason": "Utilization > 1.0"})
            continue
            
        # 4. Calculate total allowed VRAM for vLLM
        allowed_vram_gb = total_gpu_vram_gb * config["utilization"]
        
        # 5. Calculate remaining budget for KV Cache
        kv_cache_budget_gb = allowed_vram_gb - model_weights_gb - pytorch_overhead_gb
        
        if kv_cache_budget_gb <= 0:
            print(f"  Result: [FAILED] OOM on Load! Weights ({model_weights_gb:.2f}GB) exceed allowed VRAM ({allowed_vram_gb:.2f}GB).\n")
            results.append({"config": config, "status": "FAILED", "reason": "OOM on model load"})
            continue
            
        # 6. Calculate KV Cache requirement per maximum sequence
        # Formula: 2 * 30 layers * 3 kv_heads * 64 head_dim * bytes_per_param * max_len
        bytes_per_max_seq = 2 * 30 * 3 * 64 * bytes_per_param * config["max_len"]
        seq_memory_gb = bytes_per_max_seq / (1024**3)
        
        # 7. Will the engine even be able to fit ONE sequence?
        if seq_memory_gb > kv_cache_budget_gb:
            print(f"  Result: [FAILED] ValueError! max_model_len ({config['max_len']}) requires {seq_memory_gb:.2f}GB KV cache, but only {kv_cache_budget_gb:.2f}GB available.\n")
            results.append({"config": config, "status": "FAILED", "reason": "KV Cache capacity < 1 sequence"})
            continue
            
        # If we passed all checks, the engine starts successfully!
        max_concurrent_sequences = int(kv_cache_budget_gb // seq_memory_gb)
        print(f"  Result: [SUCCESS] Engine started! Weights: {model_weights_gb:.2f}GB. KV Budget: {kv_cache_budget_gb:.2f}GB. Max Concurrency: {max_concurrent_sequences} users.\n")
        results.append({"config": config, "status": "SUCCESS", "max_concurrency": max_concurrent_sequences})

    # Save to artifacts
    with open("artifacts/day30/vllm_config_matrix.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Logs saved to artifacts/day30/vllm_config_matrix.json")

if __name__ == "__main__":
    simulate_vllm_startup()
