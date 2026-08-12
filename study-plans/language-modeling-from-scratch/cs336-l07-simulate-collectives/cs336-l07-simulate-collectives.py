import torch
from typing import List


def simulate_collectives(
    rank_tensors: List[torch.Tensor], collective: str
) -> List[torch.Tensor]:
    
    if not isinstance(rank_tensors, list) or len(rank_tensors) == 0:
        raise ValueError("rank_tensors must be a nonempty list of tensors")

    R = len(rank_tensors)

    ref = rank_tensors[0]
    if ref.dim() != 1:
        raise ValueError("All rank tensors must be one-dimensional")
    length = ref.shape[0]
    dtype = ref.dtype
    device = ref.device

    for t in rank_tensors:
        if t.dim() != 1:
            raise ValueError("All rank tensors must be one-dimensional")
        if t.shape[0] != length:
            raise ValueError("All rank tensors must have equal length")
        if t.dtype != dtype:
            raise ValueError("All rank tensors must have equal dtype")
        if t.device != device:
            raise ValueError("All rank tensors must be on the same device")

    if collective == "all_gather":
        # Concatenate all rank tensors in rank order; every rank gets its own copy.
        gathered = torch.cat([t.reshape(-1).clone() for t in rank_tensors], dim=0)
        return [gathered.clone() for _ in range(R)]

    elif collective == "all_reduce":
        # Elementwise sum across ranks; every rank gets its own copy of the sum.
        stacked = torch.stack([t.reshape(-1) for t in rank_tensors], dim=0)
        summed = stacked.sum(dim=0)
        return [summed.clone() for _ in range(R)]

    elif collective == "reduce_scatter":
        if length % R != 0:
            raise ValueError(
                "Tensor length must be divisible by the rank count for reduce_scatter"
            )
        stacked = torch.stack([t.reshape(-1) for t in rank_tensors], dim=0)
        summed = stacked.sum(dim=0)
        chunks = torch.chunk(summed, R, dim=0)
        return [c.clone() for c in chunks]

    elif collective == "all_to_all":
        if length % R != 0:
            raise ValueError(
                "Tensor length must be divisible by the rank count for all_to_all"
            )
        # Split every source tensor into R chunks; destination chunk r
        # concatenates chunk r from every source, in source-rank order.
        per_source_chunks = [
            torch.chunk(t.reshape(-1), R, dim=0) for t in rank_tensors
        ]
        outputs = []
        for r in range(R):
            dest_chunk = torch.cat(
                [per_source_chunks[s][r].clone() for s in range(R)], dim=0
            )
            outputs.append(dest_chunk)
        return outputs

    else:
        raise ValueError(f"Unsupported collective: {collective!r}")