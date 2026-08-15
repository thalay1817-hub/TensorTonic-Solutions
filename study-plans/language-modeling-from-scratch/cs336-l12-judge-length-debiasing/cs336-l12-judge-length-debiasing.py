import torch

def judge_length_debiasing(num_models, model_a, model_b, length_differences, outcomes, steps, learning_rate):
    device = model_a.device
    dtype = torch.float64

    a_idx = model_a.to(device=device, dtype=torch.long)
    b_idx = model_b.to(device=device, dtype=torch.long)
    delta = length_differences.to(device=device, dtype=dtype)
    y = outcomes.to(device=device, dtype=dtype)

    N = a_idx.shape[0]

    u = torch.zeros(num_models, dtype=dtype, device=device)
    beta = torch.zeros((), dtype=dtype, device=device)

    for _ in range(steps):
        logits = u[a_idx] - u[b_idx] + beta * delta
        p = torch.sigmoid(logits)
        residual = p - y

        grad_beta = (residual * delta).mean() if N > 0 else torch.zeros((), dtype=dtype, device=device)

        grad_u = torch.zeros(num_models, dtype=dtype, device=device)
        if N > 0:
            grad_u.index_add_(0, a_idx, residual)
            grad_u.index_add_(0, b_idx, -residual)
            grad_u = grad_u / N

        u = u - learning_rate * grad_u
        beta = beta - learning_rate * grad_beta

        # Center utilities to fix the additive gauge freedom
        u = u - u.mean()

    # Recompute probabilities without the length term
    if N > 0:
        logits0 = u[a_idx] - u[b_idx]
        p0 = torch.sigmoid(logits0)
    else:
        p0 = torch.zeros(0, dtype=dtype, device=device)

    win_sum = torch.zeros(num_models, dtype=dtype, device=device)
    win_count = torch.zeros(num_models, dtype=dtype, device=device)

    if N > 0:
        win_sum.index_add_(0, a_idx, p0)
        win_count.index_add_(0, a_idx, torch.ones_like(p0))
        win_sum.index_add_(0, b_idx, 1 - p0)
        win_count.index_add_(0, b_idx, torch.ones_like(p0))

    half = torch.full((num_models,), 0.5, dtype=dtype, device=device)
    win_rates = torch.where(win_count > 0, win_sum / win_count.clamp(min=1), half)

    return {
        "length_coefficient": float(beta.item()),
        "debiased_probabilities": p0,
        "model_win_rates": win_rates,
    }