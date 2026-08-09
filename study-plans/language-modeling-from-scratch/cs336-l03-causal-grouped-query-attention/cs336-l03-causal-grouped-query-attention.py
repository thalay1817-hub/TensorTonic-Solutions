import torch
import math


def causal_gqa(q, k, v):
    orig_dtype = q.dtype
    B, Hq, S, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv

    # Cast to float32 for computation
    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)

    # Reshape queries to expose the (kv_head, group) structure without
    # repeating k or v
    q_grouped = q_f32.view(B, Hkv, G, S, D)

    # Grouped einsum: contract over head-width D, keeping kv-head, group,
    # query-seq, key-seq axes
    scores = torch.einsum('bhgid,bhjd->bhgij', q_grouped, k_f32)
    scores = scores / math.sqrt(D)

    # Causal mask: query position i can attend to key position j only if j <= i
    causal_mask = torch.triu(
        torch.ones(S, S, dtype=torch.bool, device=q.device), diagonal=1
    )
    scores = scores.masked_fill(causal_mask, float('-inf'))

    probs = torch.softmax(scores, dim=-1)

    # Weighted sum over values, still without materializing repeated v
    out_grouped = torch.einsum('bhgij,bhjd->bhgid', probs, v_f32)

    # Reshape back to (B, Hq, S, D)
    out = out_grouped.reshape(B, Hq, S, D)

    return out.to(orig_dtype)