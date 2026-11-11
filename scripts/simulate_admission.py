import torch
import json

def run_admission_simulation():
    # SmolLM-135M specifications
    num_layers = 30
    num_kv_heads = 3
    head_dim = 64
    bytes_per_element = 2 # dtype float16
    
    # Simulate a small 2GB GPU environment for this test
    TOTAL_VRAM_BYTES = 2 * 1024 * 1024 * 1024
    SAFETY_MARGIN_PERCENT = 0.20 # 20% reserved for OS and forward-pass activations
    
    # 1. Calculate usable budget
    kv_budget_bytes = TOTAL_VRAM_BYTES * (1 - SAFETY_MARGIN_PERCENT)
    
    # 2. Memory math per user
    bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * bytes_per_element
    max_context_per_user = 2048
    bytes_per_user = bytes_per_token * max_context_per_user
    
    # 3. Calculate Admission Limit
    admission_limit = int(kv_budget_bytes // bytes_per_user)
    
    print(f"--- Inference Engine Admission Controller ---")
    print(f"Total Simulated VRAM: 2.00 GB")
    print(f"Safety Margin: {SAFETY_MARGIN_PERCENT*100}%")
    print(f"Usable KV Budget: {kv_budget_bytes / (1024**3):.2f} GB")
    print(f"KV Memory per User ({max_context_per_user} tokens): {bytes_per_user / (1024**2):.2f} MB")
    print(f"-> CONCRETE ADMISSION LIMIT: {admission_limit} Concurrent Users\n")
    
    # 4. Simulate Live Web Traffic
    print("--- Simulating Incoming Web Requests ---")
    active_users = 0
    allocated_tensors = [] # We actually allocate tensors to simulate RAM usage
    
    # Try to send Admission Limit + 5 users
    for request_id in range(1, admission_limit + 6):
        if active_users < admission_limit:
            active_users += 1
            # Allocate actual zero-tensor in PyTorch to simulate KV Cache allocation
            # Shape: [layers, 2 (K&V), batch=1, kv_heads, seq_len, head_dim]
            user_cache = torch.zeros(
                (num_layers, 2, 1, num_kv_heads, max_context_per_user, head_dim),
                dtype=torch.float16
            )
            allocated_tensors.append(user_cache)
            actual_mb = (user_cache.element_size() * user_cache.nelement()) / (1024**2)
            print(f"[ACCEPT] User {request_id} admitted. Allocated {actual_mb:.2f} MB. Active: {active_users}/{admission_limit}")
        else:
            # Server rejects the request, protecting the Safety Margin
            print(f"[REJECT] User {request_id} blocked! Admission Limit ({admission_limit}) reached. OOM prevented.")
            
    # Save test logs
    log_data = {
        "budget_gb": 2.0,
        "admission_limit": admission_limit,
        "active_users_admitted": active_users
    }
    with open("artifacts/day28/admission_log.json", "w") as f:
        json.dump(log_data, f, indent=4)
    print("\nLogs saved to artifacts/day28/admission_log.json")

if __name__ == "__main__":
    run_admission_simulation()
