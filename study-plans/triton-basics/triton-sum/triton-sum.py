import torch
import triton
import triton.language as tl


@triton.jit
def sum_reduce_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
   
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

   
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)


    partial = tl.sum(x, axis=0)

    
    tl.atomic_add(out_ptr, partial)


def solve(x: torch.Tensor, out: torch.Tensor):
    n = x.numel()

   
    out.zero_()

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    sum_reduce_kernel[grid](
        x,
        out,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )