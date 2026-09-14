import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(
    x_ptr, mask_ptr, out_ptr,
    N,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_bounds = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask_bounds, other=0.0)
    m = tl.load(mask_ptr + offsets, mask=mask_bounds, other=0.0)

    scale = 1.0 / (1.0 - p)
    y = x * m * scale

    tl.store(out_ptr + offsets, y, mask=mask_bounds)


def solve(x: torch.Tensor, mask: torch.Tensor, out: torch.Tensor, p: float):
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    dropout_kernel[grid](
        x, mask, out,
        N,
        p,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out