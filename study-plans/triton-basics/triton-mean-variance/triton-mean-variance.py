import torch
import triton
import triton.language as tl


@triton.jit
def mean_var_kernel(
    x_ptr,
    sum_ptr,
    sumsq_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one contiguous tile of BLOCK_SIZE elements.
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask out lanes past N; masked lanes load as 0.0 so they
    # contribute zero to both the sum and the sum-of-squares.
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Single pass: accumulate both sum(x) and sum(x*x) for this tile.
    partial_sum = tl.sum(x, axis=0)
    partial_sumsq = tl.sum(x * x, axis=0)

    # One atomic add per statistic per program.
    tl.atomic_add(sum_ptr, partial_sum)
    tl.atomic_add(sumsq_ptr, partial_sumsq)


def solve(x: torch.Tensor, mean_out: torch.Tensor, var_out: torch.Tensor):
    n = x.numel()

    # Scratch buffers for the two reductions, zeroed before launch
    # since atomics accumulate into whatever is already there.
    sum_buf = torch.zeros(1, device=x.device, dtype=torch.float32)
    sumsq_buf = torch.zeros(1, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    mean_var_kernel[grid](
        x,
        sum_buf,
        sumsq_buf,
        n,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Finalize on the host: divide by N, apply E[x^2] - mu^2, no .item().
    mean = sum_buf / n
    var = sumsq_buf / n - mean * mean

    mean_out.copy_(mean)
    var_out.copy_(var)