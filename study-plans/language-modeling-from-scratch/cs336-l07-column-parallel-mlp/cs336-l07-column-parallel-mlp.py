import torch
from typing import List, Optional, Dict


def column_parallel_mlp(
    x: torch.Tensor,
    weight_shards: List[torch.Tensor],
    bias_shards: Optional[List[Optional[torch.Tensor]]],
) -> Dict[str, List]:
    
    if not isinstance(x, torch.Tensor):
        raise ValueError("x must be a PyTorch tensor")
    if x.dim() != 2:
        raise ValueError("x must be a rank-two tensor")
    batch, din = x.shape
    if batch <= 0 or din <= 0:
        raise ValueError("x must have positive batch and input dimensions")

    supported_dtypes = (torch.float32, torch.bfloat16)
    if x.dtype not in supported_dtypes:
        raise ValueError(f"Unsupported dtype: {x.dtype}")
    if not x.is_floating_point():
        raise ValueError("x must be a floating-point tensor")

    if not isinstance(weight_shards, list) or len(weight_shards) == 0:
        raise ValueError("weight_shards must be a nonempty list")

    R = len(weight_shards)

    if bias_shards is not None:
        if not isinstance(bias_shards, list) or len(bias_shards) != R:
            raise ValueError("bias_shards must be None or match weight_shards length")
    else:
        bias_shards = [None] * R

    device = x.device
    dtype = x.dtype

    local_activations: List[torch.Tensor] = []

    for r in range(R):
        w = weight_shards[r]
        if not isinstance(w, torch.Tensor):
            raise ValueError("Each weight shard must be a PyTorch tensor")
        if w.dim() != 2:
            raise ValueError("Each weight shard must have shape (Din, Dout_r)")
        if w.shape[0] != din:
            raise ValueError("Each weight shard must have first dimension equal to Din")
        if w.shape[1] <= 0:
            raise ValueError("Every Dout_r must be positive")
        if w.dtype != dtype or w.device != device:
            raise ValueError("All tensors must share dtype and device")

        b = bias_shards[r]
        if b is not None:
            if not isinstance(b, torch.Tensor):
                raise ValueError("Each bias shard must be a PyTorch tensor or None")
            if b.dim() != 1 or b.shape[0] != w.shape[1]:
                raise ValueError("Each bias shard must have shape (Dout_r,) matching its weight shard")
            if b.dtype != dtype or b.device != device:
                raise ValueError("All tensors must share dtype and device")

        y_r = x @ w
        if b is not None:
            y_r = y_r + b

        local_activations.append(y_r)

    full_activation = torch.cat(local_activations, dim=-1)

    return {
        "local_activations": local_activations,
        "full_activation": full_activation,
    }