# src/inference_math/sampling.py
import torch

def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Applies temperature scaling to logits."""
    if temperature == 0.0:
        return logits # Handled as greedy later
    return logits / temperature

def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Keeps only the top k tokens, sets others to -inf."""
    if top_k <= 0:
        return logits
    
    # Find the top_k values
    top_k_values, _ = torch.topk(logits, top_k)
    # The minimum value in the top_k is our threshold
    kth_best_value = top_k_values[:, -1].unsqueeze(-1)
    
    # Mask everything below the threshold with -inf
    logits = torch.where(logits < kth_best_value, torch.tensor(float('-inf')).to(logits.device), logits)
    return logits

def sample_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0) -> int:
    """Samples a single token ID from logits based on temperature and top_k."""
    # 1. Greedy Path (T=0)
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1).item()
    
    # 2. Apply Temperature
    scaled_logits = apply_temperature(logits, temperature)
    
    # 3. Apply Top-K
    filtered_logits = apply_top_k(scaled_logits, top_k)
    
    # 4. Softmax to get probabilities
    probs = torch.softmax(filtered_logits, dim=-1)
    
    # 5. Multinomial Sampling (Random draw based on probabilities)
    # torch.multinomial returns the sampled index
    sampled_id = torch.multinomial(probs, num_samples=1).item()
    return sampled_id