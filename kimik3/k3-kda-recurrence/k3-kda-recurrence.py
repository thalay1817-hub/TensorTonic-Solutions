import torch
import torch.nn.functional as F


def kda_recurrence(query, key, value, decay_logits, write_strength,
                   output_gate_logits, output_projection, initial_state,
                   g_min, eps=1e-6):
    B, S, H, Dk = query.shape
    Dv = value.shape[-1]
    dtype = query.dtype

    # write_strength may be (B, S, H) or (B, S, H, 1)
    beta = write_strength
    if beta.dim() == 4:
        beta = beta[..., 0]

    # retention vectors alpha_t = exp(g_min * sigmoid(z_t)), shape (B, S, H, Dk)
    alpha = torch.exp(g_min * torch.sigmoid(decay_logits))

    state = initial_state.clone()  # (B, H, Dk, Dv); input is never mutated
    outs = []

    for t in range(S):
        q_t = query[:, t]           # (B, H, Dk)
        k_t = key[:, t]             # (B, H, Dk)
        v_t = value[:, t]           # (B, H, Dv)
        a_t = alpha[:, t]           # (B, H, Dk)
        b_t = beta[:, t]            # (B, H)

        # Diag(alpha) S_{t-1}: broadcast alpha over the value dimension
        decayed = a_t.unsqueeze(-1) * state                       # (B, H, Dk, Dv)

        # erase term: beta * k (k^T decayed)
        kS = torch.einsum('bhk,bhkv->bhv', k_t, decayed)          # (B, H, Dv)
        erase = b_t[..., None, None] * k_t.unsqueeze(-1) * kS.unsqueeze(-2)

        # write term: beta * k v^T
        write = b_t[..., None, None] * k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

        state = decayed - erase + write

        # read from the UPDATED state: o~_t = S_t^T q_t
        o_t = torch.einsum('bhkv,bhk->bhv', state, q_t)           # (B, H, Dv)
        outs.append(o_t)

    o = torch.stack(outs, dim=1)                                  # (B, S, H, Dv)

    # RMS normalization over each head's Dv values
    o = o * torch.rsqrt(o.pow(2).mean(dim=-1, keepdim=True) + eps)

    # sigmoid output gate
    o = o * torch.sigmoid(output_gate_logits)

    # concatenate heads and project: (B, S, H*Dv) -> (B, S, Dmodel)
    o = o.reshape(B, S, H * Dv)
    outputs = F.linear(o, output_projection)

    return {
        "outputs": outputs.to(dtype),
        "final_state": state.to(dtype),
    }