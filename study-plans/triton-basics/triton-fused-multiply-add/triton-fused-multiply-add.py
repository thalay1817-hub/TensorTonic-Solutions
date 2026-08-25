import torch
import triton
import triton.language as tl


@triton.jit
def fma_kernel(
    a,
    x_ptr,
    y_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one contiguous tile of BLOCK_SIZE elements.
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask out lanes that fall past N (tail of the array).
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Single fused expression so Triton lowers this to one hardware FMA.
    out = a * x + y

    tl.store(out_ptr + offsets, out, mask=mask)


def solve(a: float, x: torch.Tensor, y: torch.Tensor, out: torch.Tensor):
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    fma_kernel[grid](
        a,
        x,
        y,
        out,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )