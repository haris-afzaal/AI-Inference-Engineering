import torch
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def measure_phases():
    print("Loading model for timing (SmolLM-135M)...")
    model_name = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "Artificial Intelligence inference engineering is the process of"
    print(f"\nPrompt: '{prompt}'")

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    prompt_length = input_ids.shape[1]

    print("\n--- Starting Generation ---")

    # Measure TTFT (Prefill)
    t0 = time.perf_counter()
    with torch.no_grad():
        out_first = model.generate(
            input_ids, max_new_tokens=1, do_sample=False
        )
    t1 = time.perf_counter()

    ttft_ms = (t1 - t0) * 1000
    first_token_id = out_first[0, -1].item()
    first_token_text = tokenizer.decode(first_token_id)

    print(f"Time To First Token (TTFT): {ttft_ms:.2f} ms | Token: '{first_token_text}'")

    # Measure ITL (Decode) over next 5 tokens
    current_ids = out_first
    decode_times = []
    generated_tokens = [first_token_text]

    for i in range(5):
        t_start = time.perf_counter()
        with torch.no_grad():
            current_ids = model.generate(
                current_ids, max_new_tokens=1, do_sample=False
            )
        t_end = time.perf_counter()

        token_ms = (t_end - t_start) * 1000
        decode_times.append(token_ms)

        new_token_id = current_ids[0, -1].item()
        new_token_text = tokenizer.decode(new_token_id)
        generated_tokens.append(new_token_text)

        print(f"ITL Token {i+1}: {token_ms:.2f} ms | Token: '{new_token_text}'")

    avg_itl_ms = sum(decode_times) / len(decode_times)
    print(f"\nAverage Inter-Token Latency (ITL): {avg_itl_ms:.2f} ms")

    # Save log
    log_data = {
        "prompt_length_tokens": prompt_length,
        "ttft_ms": ttft_ms,
        "decode_times_ms": decode_times,
        "avg_itl_ms": avg_itl_ms,
        "generated_text": "".join(generated_tokens)
    }

    with open("artifacts/day22/day22-prefill-decode.json", "w") as f:
        json.dump(log_data, f, indent=4)

    print("Saved to artifacts/day22/day22-prefill-decode.json")

if __name__ == "__main__":
    measure_phases()
