import torch
import triton
import triton.language as tl


@triton.jit
def l2_norm_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sq = x * x
    partial_sum = tl.sum(sq, axis=0)

    tl.atomic_add(out_ptr, partial_sum)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
   
    assert x.is_cuda and out.is_cuda
    assert x.dtype == torch.float32
    N = x.numel()

    # Scratch buffer to accumulate sum of squares across programs.
    sumsq_buf = torch.zeros(1, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    l2_norm_kernel[grid](
        x,
        sumsq_buf,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Square root applied once on the host.
    out.copy_(torch.sqrt(sumsq_buf))