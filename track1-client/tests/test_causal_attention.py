import pytest
import torch
import math
import torch.nn.functional as F

def test_causal_mask():
    batch_size = 2
    num_heads = 4
    seq_len = 4
    d_k = 16

    # Q, K projections
    Q = torch.randn(batch_size, num_heads, seq_len, d_k)
    K = torch.randn(batch_size, num_heads, seq_len, d_k)

    # Scaled dot-product attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Lower triangular causal mask (1s where visible, 0s where masked out)
    mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)

    # Mask future positions with -inf
    masked_scores = scores.masked_fill(mask == 0, float('-inf'))

    # Softmax over last dimension
    weights = F.softmax(masked_scores, dim=-1)

    # Assert shape matches
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

    # Verify no future leakage: upper triangular elements (where j > i) must be exactly 0.0
    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    assert weights[b, h, i, j].item() == 0.0, f"Future leakage at ({i}, {j})"

    # Verify rows sum to 1.0 (probabilities across valid tokens sum to 1)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    print("\n✓ Causal Attention Test Passed Successfully!")

if __name__ == "__main__":
    test_causal_mask()
