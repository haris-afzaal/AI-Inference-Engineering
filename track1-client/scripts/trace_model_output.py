import torch
import time
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

def run_model_trace():
    print("Loading tokenizer and model (SmolLM-135M)...")

    model_name = "HuggingFaceTB/SmolLM-135M"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "The most important rule of inference engineering is"

    print(f"\nPrompt: '{prompt}'")

    inputs = tokenizer(prompt, return_tensors="pt")

    start_time = time.time()

    with torch.no_grad():
        outputs = model(**inputs)

    inference_time = time.time() - start_time

    logits = outputs.logits

    next_token_logits = logits[0, -1, :]

    probabilities = torch.softmax(next_token_logits, dim=-1)

    next_token_id = torch.argmax(probabilities).item()

    next_token = tokenizer.decode([next_token_id])

    print("\n--- MODEL TRACE ---")
    print("Logits shape:", logits.shape)
    print("Next token:", next_token)
    print("Next token ID:", next_token_id)
    print("Probability:", probabilities[next_token_id].item())
    print("Inference time:", round(inference_time, 4), "seconds")

    top_values, top_indices = torch.topk(probabilities, 5)

    print("\nTop 5 predictions:")

    for value, index in zip(top_values, top_indices):
        token = tokenizer.decode([index.item()])
        print(f"{token!r}: {value.item():.4f}")

    result = {
        "prompt": prompt,
        "next_token": next_token,
        "next_token_id": next_token_id,
        "probability": probabilities[next_token_id].item(),
        "inference_time_seconds": inference_time,
        "logits_shape": list(logits.shape)
    }

    with open("artifacts/day20/model_trace.json", "w") as f:
        json.dump(result, f, indent=4)

    print("\nTrace saved to artifacts/day20/model_trace.json")


if __name__ == "__main__":
    run_model_trace()
