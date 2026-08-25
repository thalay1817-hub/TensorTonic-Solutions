import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    x_ptr,
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

    # Branch-free clamp: elementwise maximum against zero.
    out = tl.maximum(x, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor):
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    relu_kernel[grid](
        x,
        out,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )