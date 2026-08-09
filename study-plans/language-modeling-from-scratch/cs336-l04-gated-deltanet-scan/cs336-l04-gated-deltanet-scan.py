import torch


def gated_deltanet_scan(q, k, v, gamma, beta):
    orig_dtype = q.dtype
    device = q.device

    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)
    gamma_f32 = gamma.to(torch.float32)
    beta_f32 = beta.to(torch.float32)

    B, S, Dk = q_f32.shape
    Dv = v_f32.shape[-1]

    state = torch.zeros(B, Dk, Dv, dtype=torch.float32, device=device)
    identity = torch.eye(Dk, dtype=torch.float32, device=device).unsqueeze(0).expand(B, Dk, Dk)

    outputs = []
    for t in range(S):
        gt = gamma_f32[:, t].view(B, 1, 1)     # (B,1,1)
        bt = beta_f32[:, t].view(B, 1, 1)      # (B,1,1)
        kt = k_f32[:, t, :]                     # (B, Dk)
        vt = v_f32[:, t, :]                     # (B, Dv)
        qt = q_f32[:, t, :]                     # (B, Dk)

        kk = torch.bmm(kt.unsqueeze(2), kt.unsqueeze(1))  # (B, Dk, Dk) = k k^T
        erase = identity - bt * kk                         # (B, Dk, Dk)

        erased_state = torch.bmm(erase, state)              # (I - beta*k k^T) S_{t-1}

        kv = torch.bmm(kt.unsqueeze(2), vt.unsqueeze(1))    # (B, Dk, Dv) = k v^T

        state = gt * erased_state + bt * kv

        yt = torch.bmm(qt.unsqueeze(1), state).squeeze(1)   # (B, Dv)
        outputs.append(yt)

    outputs_f32 = torch.stack(outputs, dim=1)
    final_state_f32 = state

    return {
        "outputs": outputs_f32.to(orig_dtype),
        "final_state": final_state_f32.to(orig_dtype),
    }