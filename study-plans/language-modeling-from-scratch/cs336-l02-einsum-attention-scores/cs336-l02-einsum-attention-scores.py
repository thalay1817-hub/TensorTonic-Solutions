import torch


def attention_scores(q, k, num_heads):
    B, S_q, D = q.shape
    _, S_k, _ = k.shape
    H = num_heads
    d_h = D // H

    # Reshape (B, S, D) -> (B, S, H, d_h) -> (B, H, S, d_h)
    q_heads = q.reshape(B, S_q, H, d_h).permute(0, 2, 1, 3)
    k_heads = k.reshape(B, S_k, H, d_h).permute(0, 2, 1, 3)

    # Contract over the head-dimension axis for every (batch, head, query, key) combo
    scores = torch.einsum('bhir,bhjr->bhij', q_heads, k_heads)

    # Scale by sqrt(d_h)
    scale = torch.sqrt(torch.tensor(d_h, dtype=q.dtype, device=q.device))
    scores = scores / scale

    return scores