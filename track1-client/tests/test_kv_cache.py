import torch
import torch.nn.functional as F
import json
import os

torch.manual_seed(42)

# Toy dimensions
d_model = 16
d_k = 16

# Random projection weights for Q, K, V
W_q = torch.randn(d_model, d_k)
W_k = torch.randn(d_model, d_k)
W_v = torch.randn(d_model, d_k)

def uncached_attention(x):
    """Standard causal attention processing all tokens at once (O(N^2))."""
    seq_len = x.shape[1]
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v
    
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)
    
    # Causal mask so future tokens can't be seen
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    
    attn = F.softmax(scores, dim=-1)
    return attn @ V

def cached_attention(x_new, past_k=None, past_v=None):
    """Processes only the NEW token, reusing past_k and past_v."""
    # Compute Q, K, V for just the new token (seq_len = 1)
    q_new = x_new @ W_q
    k_new = x_new @ W_k
    v_new = x_new @ W_v
    
    # Append new K, V to the cache
    if past_k is not None:
        K = torch.cat([past_k, k_new], dim=1)
        V = torch.cat([past_v, v_new], dim=1)
    else:
        K = k_new
        V = v_new
        
    # Attention for the new token against ALL keys (past + present)
    # q_new shape: [batch, 1, d_k], K shape: [batch, seq_len, d_k]
    scores = q_new @ K.transpose(-2, -1) / (d_k ** 0.5)
    
    # No causal mask needed here because we only have 1 query looking at past+present
    attn = F.softmax(scores, dim=-1)
    out_new = attn @ V
    
    return out_new, K, V

def run_kv_cache_test():
    print("--- Running KV Cache Correctness Test ---")
    batch_size = 1
    seq_len = 3
    
    # Random embedding for 3 tokens
    x_full = torch.randn(batch_size, seq_len, d_model)
    
    # 1. Uncached Forward Pass
    print("\n1. Running Uncached Attention (Full Sequence)...")
    out_uncached = uncached_attention(x_full)
    # Extract just the last token's output
    uncached_last_token_out = out_uncached[:, -1:, :]
    
    # 2. Cached Forward Pass
    print("2. Running Cached Attention (Token by Token)...")
    past_k, past_v = None, None
    out_cached = None
    
    for i in range(seq_len):
        x_step = x_full[:, i:i+1, :] # Feed 1 token at a time
        out_cached, past_k, past_v = cached_attention(x_step, past_k, past_v)
        print(f"   Step {i+1}: KV Cache Size -> [batch=1, seq={past_k.shape[1]}, d_k={d_k}]")
        
    # 3. Verify Correctness
    difference = (uncached_last_token_out - out_cached).abs().max().item()
    
    print("\n--- Verification ---")
    print(f"Max difference between Uncached and Cached: {difference:.8f}")
    
    is_correct = difference < 1e-5
    if is_correct:
        print("[PASSED] Cached output exactly matches uncached output!")
    else:
        print("[FAILED] Outputs do not match!")
        
    # Save verification log
    os.makedirs("artifacts/day24", exist_ok=True)
    log_data = {
        "max_difference": difference,
        "is_correct": is_correct,
        "final_cache_shape": list(past_k.shape)
    }
    with open("artifacts/day24/day24-kv-cache.json", "w") as f:
        json.dump(log_data, f, indent=4)
    print("\nLogs saved to artifacts/day24/day24-kv-cache.json")

if __name__ == "__main__":
    run_kv_cache_test()
