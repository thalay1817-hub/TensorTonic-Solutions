import torch

def grpo_length_normalization_audit(token_log_probs, advantages, token_mask):
    dtype = token_log_probs.dtype
    device = token_log_probs.device

    mask_f = token_mask.to(dtype)

    # Masked token sums and active counts per row
    masked_log_probs = token_log_probs * mask_f
    token_sums = masked_log_probs.sum(dim=1)
    counts = mask_f.sum(dim=1)

    sequence_summed = advantages.to(dtype) * token_sums

    token_averaged = torch.where(
        counts > 0,
        sequence_summed / counts.clamp(min=1),
        torch.zeros_like(sequence_summed),
    ).to(dtype)

    numerator = sequence_summed.abs().sum()
    denominator = token_averaged.abs().sum()

    if denominator.item() == 0:
        magnitude_ratio = torch.zeros((), dtype=dtype, device=device)
    else:
        magnitude_ratio = numerator / denominator

    return {
        "sequence_summed": sequence_summed,
        "token_averaged": token_averaged,
        "magnitude_ratio": magnitude_ratio.to(dtype),
    }