# tests/test_sampling.py
import torch
import json
import os
import sys

# Ensure src is in the path to import our module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference_math.sampling import sample_token, apply_top_k

def test_greedy_is_deterministic():
    print("--- Running Greedy (T=0) Test ---")
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.5]]) # Max is at index 1
    
    sampled_id_1 = sample_token(logits, temperature=0.0)
    sampled_id_2 = sample_token(logits, temperature=0.0)
    
    assert sampled_id_1 == 1, f"Expected index 1, got {sampled_id_1}"
    assert sampled_id_1 == sampled_id_2, "FAIL: Greedy search is not deterministic!"
    print("PASSED: Greedy search is deterministic and selected the highest logit.\n")

def test_top_k_filtering():
    print("--- Running Top-K Filtering Test ---")
    logits = torch.tensor([[10.0, 8.0, 3.0, 1.0, 0.5]])
    
    # Top 2 are index 0 (10.0) and index 1 (8.0)
    filtered = apply_top_k(logits, top_k=2)
    
    # Check if values outside top 2 are masked with -inf
    assert filtered[0, 2] == float('-inf'), "FAIL: Index 2 should be masked"
    assert filtered[0, 3] == float('-inf'), "FAIL: Index 3 should be masked"
    assert filtered[0, 4] == float('-inf'), "FAIL: Index 4 should be masked"
    assert filtered[0, 0] == 10.0, "FAIL: Top values should remain unchanged"
    print("PASSED: Top-K correctly masked lower values to -inf.\n")

def test_temperature_scaling():
    print("--- Running Temperature Distribution Test ---")
    torch.manual_seed(42)
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]]) # Max is at index 3
    
    # Sample with low temperature (Cold / Confident)
    samples_low_temp = [sample_token(logits, temperature=0.1) for _ in range(100)]
    
    # Sample with high temperature (Hot / Creative)
    samples_high_temp = [sample_token(logits, temperature=10.0) for _ in range(100)]
    
    # Low temp should almost always pick index 3 (the highest logit)
    assert samples_low_temp.count(3) > 95, "FAIL: Low temperature was not confident enough"
    
    # High temp should have a wider spread, so index 3 shouldn't be picked 90+ times
    assert samples_high_temp.count(3) < 90, "FAIL: High temperature did not spread the distribution"
    print(f"PASSED: Low temp picked index 3 [{samples_low_temp.count(3)}/100] times.")
    print(f"PASSED: High temp picked index 3 [{samples_high_temp.count(3)}/100] times.\n")

def save_log():
    log_data = {
        "greedy_deterministic": True,
        "top_k_works": True,
        "temperature_controls_spread": True
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/day19-sampling.json", "w") as f:
        json.dump(log_data, f, indent=4)
    print("[✔] Logs saved to logs/day19-sampling.json")

if __name__ == "__main__":
    test_greedy_is_deterministic()
    test_top_k_filtering()
    test_temperature_scaling()
    save_log()