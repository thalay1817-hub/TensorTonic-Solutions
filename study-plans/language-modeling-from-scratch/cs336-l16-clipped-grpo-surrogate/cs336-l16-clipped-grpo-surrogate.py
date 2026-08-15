import torch

def clipped_grpo_surrogate(new_log_probs, old_log_probs, advantages, token_mask, clip_epsilon):
    dtype = new_log_probs.dtype
    device = new_log_probs.device

    batch, seq_len = new_log_probs.shape

    log_ratio = new_log_probs - old_log_probs
    clamped_log_ratio = torch.clamp(log_ratio, min=-60.0, max=60.0)
    ratio = torch.exp(clamped_log_ratio)

    clipped_ratio = torch.clamp(ratio, min=1.0 - clip_epsilon, max=1.0 + clip_epsilon)

    adv_expanded = advantages.unsqueeze(1).to(dtype)

    unclipped_term = ratio * adv_expanded
    clipped_term = clipped_ratio * adv_expanded

    surrogate = torch.minimum(unclipped_term, clipped_term)

    mask_f = token_mask.to(dtype)

    # Global masked mean for loss
    total_active = mask_f.sum()
    if total_active.item() == 0:
        loss = torch.zeros((), dtype=dtype, device=device)
    else:
        loss = -(surrogate * mask_f).sum() / total_active

    # Per-sequence masked mean
    per_seq_active = mask_f.sum(dim=1)
    per_seq_sum = (surrogate * mask_f).sum(dim=1)
    per_sequence_surrogate = torch.where(
        per_seq_active > 0,
        per_seq_sum / per_seq_active.clamp(min=1),
        torch.zeros_like(per_seq_sum),
    ).to(dtype)

    # Clipped fraction: active tokens where ratio strictly outside [1-eps, 1+eps]
    outside = (ratio < (1.0 - clip_epsilon)) | (ratio > (1.0 + clip_epsilon))
    outside_f = outside.to(dtype)

    if total_active.item() == 0:
        clipped_fraction = torch.zeros((), dtype=dtype, device=device)
    else:
        clipped_fraction = (outside_f * mask_f).sum() / total_active

    return {
        "loss": loss.to(dtype),
        "per_sequence_surrogate": per_sequence_surrogate,
        "clipped_fraction": clipped_fraction.to(dtype),
    }