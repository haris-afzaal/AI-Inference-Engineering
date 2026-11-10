import json

def calculate_kv_memory():
    # Specifications for two different models
    models = {
        "SmolLM-135M": {
            "num_layers": 30,
            "num_kv_heads": 3,
            "head_dim": 64,
            "bytes_per_element": 2, # FP16 / BF16
        },
        "Llama-3-8B": {
            "num_layers": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "bytes_per_element": 2, # FP16 / BF16
        }
    }
    
    # 8 GB reserved for KV cache (out of maybe 16GB or 24GB total GPU memory)
    # The rest is used by model weights and PyTorch overhead
    gpu_kv_budget_bytes = 8 * 1024 * 1024 * 1024 
    
    results = {}
    
    print(f"--- KV Cache Memory Calculator ---")
    print(f"GPU Budget for KV Cache: 8.00 GB\n")

    for model_name, config in models.items():
        # THE FORMULA
        # 2 (for K and V) * layers * kv_heads * head_dim * bytes_per_param
        bytes_per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * config["bytes_per_element"]
        
        # Calculate for 1000 tokens (a typical prompt)
        mb_per_1k_tokens = (bytes_per_token * 1000) / (1024 * 1024)
        
        # Max tokens that can fit in our 8GB budget
        max_tokens_capacity = gpu_kv_budget_bytes // bytes_per_token
        
        # How many concurrent users (batch size) if each user has a 2048 token context?
        concurrent_users_2k_context = max_tokens_capacity // 2048
        
        results[model_name] = {
            "bytes_per_token": bytes_per_token,
            "mb_per_1k_tokens": round(mb_per_1k_tokens, 2),
            "max_tokens_in_8gb": max_tokens_capacity,
            "concurrent_users_at_2048_ctx": concurrent_users_2k_context
        }
        
        print(f"Model: {model_name}")
        print(f"  -> Memory per token: {bytes_per_token} Bytes")
        print(f"  -> Memory per 1K tokens: {mb_per_1k_tokens:.2f} MB")
        print(f"  -> Max Total Tokens in 8GB: {max_tokens_capacity:,}")
        print(f"  -> Max Concurrent Users (2K Context each): {concurrent_users_2k_context} users\n")

    # Save to artifacts
    with open("artifacts/day25/kv-capacity.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Results saved to artifacts/day25/kv-capacity.json")

if __name__ == "__main__":
    calculate_kv_memory()
