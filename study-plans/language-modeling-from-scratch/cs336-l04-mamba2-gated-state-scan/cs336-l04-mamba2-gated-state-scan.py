import torch


def mamba2_gated_scan(q, k, v, gamma):
    orig_dtype = q.dtype
    device = q.device

    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)
    gamma_f32 = gamma.to(torch.float32)

    B, S, Dk = q_f32.shape
    Dv = v_f32.shape[-1]

    state = torch.zeros(B, Dk, Dv, dtype=torch.float32, device=device)
    outputs = []

    for t in range(S):
        gt = gamma_f32[:, t].view(B, 1, 1)      # (B, 1, 1)
        kt = k_f32[:, t, :]                      # (B, Dk)
        vt = v_f32[:, t, :]                       # (B, Dv)
        qt = q_f32[:, t, :]                        # (B, Dk)

        outer = torch.bmm(kt.unsqueeze(2), vt.unsqueeze(1))  # (B, Dk, Dv)
        state = gt * state + outer

        yt = torch.bmm(qt.unsqueeze(1), state).squeeze(1)   # (B, Dv)
        outputs.append(yt)

    outputs_f32 = torch.stack(outputs, dim=1)  # (B, S, Dv)
    final_state_f32 = state

    return {
        "outputs": outputs_f32.to(orig_dtype),
        "final_state": final_state_f32.to(orig_dtype),
    }