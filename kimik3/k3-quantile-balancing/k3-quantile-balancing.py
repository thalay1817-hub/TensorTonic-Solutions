import torch


def quantile_balancing(router_scores, current_bias, selected_count):
    s = torch.as_tensor(router_scores)
    if not s.is_floating_point():
        s = s.to(torch.get_default_dtype())
    m, n = s.shape
    k = int(selected_count)

    bias = torch.as_tensor(current_bias).to(device=s.device, dtype=s.dtype)
    biased = s + bias.unsqueeze(0)

    # top k+1 biased scores
    top_vals, top_idx = torch.topk(biased, k + 1, dim=1)
    selected_experts = top_idx[:, :k]
    alpha = top_vals[:, k]

    # mixture weights from raw scores
    selected_raw = torch.gather(s, 1, selected_experts)
    mixture_weights = selected_raw / selected_raw.sum(dim=1, keepdim=True)

    # expert loads
    expert_loads = torch.bincount(selected_experts.reshape(-1), minlength=n)

    # next bias
    q = (m * k) // n
    margins = s - alpha.unsqueeze(1)
    sorted_margins, _ = torch.sort(margins, dim=0, descending=True)
    next_bias = -sorted_margins[q]
    next_bias = next_bias - next_bias.mean()

    return {
        "selected_experts": selected_experts,
        "mixture_weights": mixture_weights,
        "expert_loads": expert_loads,
        "next_bias": next_bias,
    }