# src/inference_math/attention_reference.py
import torch
import math

def reference_attention(q, k, v, is_causal=False):
    """
    Computes Scaled Dot-Product Attention.
    Inputs shape: (batch_size, num_heads, seq_len, head_dim)
    """
    # Q * K^T
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply Causal Mask if required
    if is_causal:
        seq_len = q.size(2)
        # Create a boolean upper triangular mask (True where j > i)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1)
        # Apply -inf to future positions
        scores.masked_fill_(causal_mask, float('-inf'))
        
    # Softmax (Max-subtraction is handled internally by PyTorch's softmax for stability)
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Multiply with V
    output = torch.matmul(attn_weights, v)
    return output