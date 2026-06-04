import torch
def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    counts = counts.float()
    N = counts.sum()
    f = counts / N
    return torch.clamp((t / f).sqrt(), max=1.0)
