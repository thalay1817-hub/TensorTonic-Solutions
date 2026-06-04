import torch
def noise_distribution(counts: torch.Tensor, alpha: float = 0.75) -> torch.Tensor:
    w = torch.as_tensor(counts, dtype=torch.float64) ** alpha
    return w / w.sum()