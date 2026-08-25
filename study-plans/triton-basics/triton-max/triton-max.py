import torch
import triton
import triton.language as tl


@triton.jit
def max_reduce_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Single program covers the whole array in one tile.
    offsets = tl.arange(0, BLOCK_SIZE)

    # Mask out lanes past N; masked lanes load as -inf so they never
    # win the max comparison against real elements.
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=-float("inf"))

    # Reduce the whole tile in registers.
    result = tl.max(x, axis=0)

    # No atomics needed - single program, direct store.
    tl.store(out_ptr, result)


def solve(x: torch.Tensor, out: torch.Tensor):
    n = x.numel()

    # Block size = next power of two >= N so the whole array fits in one tile.
    BLOCK_SIZE = triton.next_power_of_2(n)

    grid = (1,)

    max_reduce_kernel[grid](
        x,
        out,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )