import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def manual_autoregressive_loop():
    model_name = "HuggingFaceTB/SmolLM-135M"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # A prompt that usually completes quickly
    prompt = "The capital of France is"
    print(f"\nPrompt: '{prompt}'")
    
    # 1. Tokenize input
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    
    # Identify the EOS token ID for this specific model
    eos_token_id = tokenizer.eos_token_id
    print(f"EOS Token ID for this model: {eos_token_id}")
    
    # Our two guards
    max_new_tokens = 20
    generated_tokens = 0
    
    print("\n--- Starting Manual Autoregressive Loop ---")
    
    while generated_tokens < max_new_tokens:
        # Step A: Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
            
        # Step B: Get logits for the LAST token only
        next_token_logits = outputs.logits[0, -1, :]
        
        # Step C: Greedy Search (Argmax) to get the predicted ID
        # We add unsqueeze to make its shape [1, 1] so it can be concatenated
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
        
        next_word = tokenizer.decode(next_token_id[0])
        print(f"Step {generated_tokens + 1}: Generated ID {next_token_id.item()} -> '{next_word}'")
        
        # GUARD 1: Check for EOS Token
        if next_token_id.item() == eos_token_id:
            print("\n[STOP] EOS Token detected! The model naturally finished its sentence.")
            break
            
        # Step D: Append the new token to the input sequence (Output = Input)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        generated_tokens += 1
        
    # GUARD 2: Check for Max Length
    if generated_tokens == max_new_tokens:
        print(f"\n[STOP] Max tokens ({max_new_tokens}) reached! Emergency stop triggered.")
        
    final_text = tokenizer.decode(input_ids[0])
    print(f"\nFinal Output: '{final_text}'")

if __name__ == "__main__":
    manual_autoregressive_loop()
