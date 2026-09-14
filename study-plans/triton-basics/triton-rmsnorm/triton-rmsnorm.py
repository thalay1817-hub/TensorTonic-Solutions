import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr, gamma_ptr, out_ptr,
    M, N,
    stride_m,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    row_ptr = x_ptr + row * stride_m
    x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Single-pass sum of squares
    sum_sq = tl.sum(x * x, axis=0)
    mean_sq = sum_sq / N
    inv_rms = tl.math.rsqrt(mean_sq + eps)

    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    y = x * inv_rms * gamma

    out_row_ptr = out_ptr + row * stride_m
    tl.store(out_row_ptr + col_offsets, y, mask=mask)


def solve(x: torch.Tensor, gamma: torch.Tensor, out: torch.Tensor, eps: float):
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    rmsnorm_kernel[grid](
        x, gamma, out,
        M, N,
        x.stride(0),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out