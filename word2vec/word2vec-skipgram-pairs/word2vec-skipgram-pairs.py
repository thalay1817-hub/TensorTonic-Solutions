import torch
def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    n = len(token_ids)
    pairs = []
    for i in range(n):
        for j in range(max(0, i - window), min(n, i + window + 1)):
            if j != i:
                pairs.append([token_ids[i].item(), token_ids[j].item()])
    if not pairs:
        return torch.zeros((0, 2), dtype=torch.int64)
    return torch.tensor(pairs, dtype=torch.int64)