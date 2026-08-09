import torch


def linear_attention_duality(q, k, v):
    orig_dtype = q.dtype
    device = q.device

    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)

    B, S, Dk = q_f32.shape
    Dv = v_f32.shape[-1]

    # ---------- Parallel form ----------
    # scores[b, i, j] = q_i . k_j
    scores = torch.einsum('bid,bjd->bij', q_f32, k_f32)

    # Causal mask: only keep j <= i (inclusive)
    causal_mask = torch.triu(
        torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1
    )
    scores = scores.masked_fill(causal_mask, 0.0)

    parallel_out_f32 = torch.einsum('bij,bjd->bid', scores, v_f32)

    # ---------- Recurrent form ----------
    state = torch.zeros(B, Dk, Dv, dtype=torch.float32, device=device)
    recurrent_outputs = []
    for t in range(S):
        kt = k_f32[:, t, :]          # (B, Dk)
        vt = v_f32[:, t, :]          # (B, Dv)
        qt = q_f32[:, t, :]          # (B, Dk)

        outer = torch.einsum('bd,be->bde', kt, vt)  # (B, Dk, Dv)
        state = state + outer

        yt = torch.einsum('bd,bde->be', qt, state)  # (B, Dv)
        recurrent_outputs.append(yt)

    recurrent_out_f32 = torch.stack(recurrent_outputs, dim=1)  # (B, S, Dv)

    final_state_f32 = state

    return {
        "parallel_output": parallel_out_f32.to(orig_dtype),
        "recurrent_output": recurrent_out_f32.to(orig_dtype),
        "final_state": final_state_f32.to(orig_dtype),
    }