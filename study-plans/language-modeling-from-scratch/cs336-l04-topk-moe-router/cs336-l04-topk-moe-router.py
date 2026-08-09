import torch


def topk_moe_router(x, w_router, expert_outputs, shared_output, k):
    orig_dtype = x.dtype
    device = x.device

    x_f32 = x.to(torch.float32)
    w_f32 = w_router.to(torch.float32)
    expert_outputs_f32 = expert_outputs.to(torch.float32)
    shared_f32 = shared_output.to(torch.float32)

    T, D = x_f32.shape
    E = w_f32.shape[-1]
    Dout = shared_f32.shape[-1]

    # Router scores (T, E)
    scores = x_f32 @ w_f32

    # Stable descending sort => ties keep lower index first
    sorted_idx = torch.argsort(scores, dim=-1, descending=True, stable=True)  # (T, E)
    selected_experts = sorted_idx[:, :k].contiguous()  # (T, k), long

    # Gather selected scores
    selected_scores = torch.gather(scores, 1, selected_experts)  # (T, k)

    # Softmax only over selected scores
    selected_weights = torch.softmax(selected_scores, dim=-1)  # (T, k)

    # Gather selected expert outputs: expert_outputs_f32 (T, E, Dout)
    idx_expanded = selected_experts.unsqueeze(-1).expand(T, k, Dout)  # (T, k, Dout)
    gathered_outputs = torch.gather(expert_outputs_f32, 1, idx_expanded)  # (T, k, Dout)

    # Weighted sum
    weighted = (selected_weights.unsqueeze(-1) * gathered_outputs).sum(dim=1)  # (T, Dout)

    output_f32 = shared_f32 + weighted

    # token_fraction: fraction of tokens selecting each expert
    selection_mask = torch.zeros(T, E, dtype=torch.float32, device=device)
    selection_mask.scatter_(1, selected_experts, 1.0)
    token_fraction_f32 = selection_mask.mean(dim=0)  # (E,)

    # probability_mass: mean of full dense softmax over all experts
    dense_probs = torch.softmax(scores, dim=-1)  # (T, E)
    probability_mass_f32 = dense_probs.mean(dim=0)  # (E,)

    return {
        "output": output_f32.to(orig_dtype),
        "selected_experts": selected_experts.to(torch.long),
        "selected_weights": selected_weights.to(orig_dtype),
        "token_fraction": token_fraction_f32.to(orig_dtype),
        "probability_mass": probability_mass_f32.to(orig_dtype),
    }