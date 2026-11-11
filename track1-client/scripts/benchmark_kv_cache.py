import torch
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def benchmark():
    model_name = "HuggingFaceTB/SmolLM-135M"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "Artificial Intelligence inference engineering is the process of"
    print(f"\nPrompt: '{prompt}'")
    
    input_ids_orig = tokenizer(prompt, return_tensors="pt")["input_ids"]
    eos_token_id = tokenizer.eos_token_id
    max_new_tokens = 20
    
    results = {}

    # --- 1. UNCACHED RUN ---
    print("\n--- Running UNCACHED ---")
    input_ids = input_ids_orig.clone()
    uncached_decode_times = []
    generated_text = ""
    
    # TTFT
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids)
    t1 = time.perf_counter()
    
    results["uncached_ttft_ms"] = (t1 - t0) * 1000
    
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
    input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    generated_text += tokenizer.decode(next_token_id[0])
    
    # Decode Loop
    for _ in range(max_new_tokens - 1):
        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids)
        t_end = time.perf_counter()
        
        uncached_decode_times.append((t_end - t_start) * 1000)
        
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        generated_text += tokenizer.decode(next_token_id[0])
        
        if next_token_id.item() == eos_token_id:
            break

    results["uncached_avg_itl_ms"] = sum(uncached_decode_times) / len(uncached_decode_times)
    print(f"Uncached TTFT: {results['uncached_ttft_ms']:.2f} ms")
    print(f"Uncached Avg ITL: {results['uncached_avg_itl_ms']:.2f} ms")

    # --- 2. CACHED RUN ---
    print("\n--- Running CACHED ---")
    input_ids = input_ids_orig.clone()
    cached_decode_times = []
    generated_text_cached = ""
    past_kv = None
    
    # TTFT (Process full prompt to generate initial cache)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    t1 = time.perf_counter()
    
    results["cached_ttft_ms"] = (t1 - t0) * 1000
    past_kv = outputs.past_key_values
    
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
    generated_text_cached += tokenizer.decode(next_token_id[0])
    
    # Decode Loop
    for _ in range(max_new_tokens - 1):
        # We only pass the NEW token, and the past_kv cache!
        input_ids = next_token_id 
        
        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_ids, past_key_values=past_kv, use_cache=True)
        t_end = time.perf_counter()
        
        cached_decode_times.append((t_end - t_start) * 1000)
        past_kv = outputs.past_key_values
        
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(-1)
        generated_text_cached += tokenizer.decode(next_token_id[0])
        
        if next_token_id.item() == eos_token_id:
            break

    results["cached_avg_itl_ms"] = sum(cached_decode_times) / len(cached_decode_times)
    results["itl_speedup_percent"] = ((results["uncached_avg_itl_ms"] - results["cached_avg_itl_ms"]) / results["uncached_avg_itl_ms"]) * 100
    results["output_match"] = (generated_text == generated_text_cached)

    print(f"Cached TTFT: {results['cached_ttft_ms']:.2f} ms")
    print(f"Cached Avg ITL: {results['cached_avg_itl_ms']:.2f} ms")
    print(f"\nOutputs Match: {results['output_match']}")
    print(f"Speedup in Decode (ITL): {results['itl_speedup_percent']:.2f}% FASTER")

    with open("artifacts/day26/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    benchmark()
