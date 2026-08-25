import torch
import triton
import triton.language as tl


INV_SQRT2: tl.constexpr = 0.7071067811865475


@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
   
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)

  
    erf_arg = x * INV_SQRT2
    out = 0.5 * x * (1.0 + tl.math.erf(erf_arg))

    tl.store(out_ptr + offsets, out, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor):
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    gelu_kernel[grid](
        x,
        out,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )