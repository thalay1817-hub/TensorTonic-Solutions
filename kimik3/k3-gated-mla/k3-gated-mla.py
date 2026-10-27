import math
import torch
import torch.nn.functional as F


def gated_mla(hidden_states, query_projection, latent_down_projection,
              key_up_projection, value_up_projection, output_gate_projection,
              output_projection, num_heads, causal):
    X = hidden_states
    B, S, D = X.shape
    H = num_heads
    Dh = D // H

    # Projections (weights are stored as (out_features, in_features))
    Q = F.linear(X, query_projection)                  # (B, S, D)
    C = F.linear(X, latent_down_projection)            # (B, S, L), the cache
    K = F.linear(C, key_up_projection)                 # (B, S, D)
    V = F.linear(C, value_up_projection)               # (B, S, D)

    # Split into heads: (B, H, S, Dh)
    def split(t):
        return t.reshape(B, S, H, Dh).transpose(1, 2)

    Qh, Kh, Vh = split(Q), split(K), split(V)

    # Scaled dot-product attention scores (no positional encoding)
    scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(Dh)  # (B, H, S, S)

    if causal:
        mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=X.device), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))

    A = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(A, Vh)                          # (B, H, S, Dh)

    # Merge heads back to model width
    O = ctx.transpose(1, 2).reshape(B, S, D)

    # Channel gate from the hidden states, then output projection
    gate = torch.sigmoid(F.linear(X, output_gate_projection))
    Y = F.linear(gate * O, output_projection)

    return {
        "output": Y.to(X.dtype),
        "latent_cache": C.to(X.dtype),
    }