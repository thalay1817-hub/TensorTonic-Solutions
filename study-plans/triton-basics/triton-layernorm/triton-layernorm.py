import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    M, N,
    stride_row,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    row_ptr = x_ptr + row * stride_row
    x = tl.load(row_ptr + col_offsets, mask=mask, other=0.0)

    # Mean over the valid (masked) lanes.
    mean = tl.sum(x, axis=0) / N

    # Center, zeroing masked lanes so they don't contribute (-mean)^2.
    centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / N

    inv_std = tl.math.rsqrt(var + eps)
    x_hat = centered * inv_std

    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=0.0)
    beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)

    y = x_hat * gamma + beta

    out_row_ptr = out_ptr + row * stride_row
    tl.store(out_row_ptr + col_offsets, y, mask=mask)


def solve(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    eps: float,
) -> None:
    
    assert x.is_cuda and gamma.is_cuda and beta.is_cuda and out.is_cuda
    assert x.dtype == torch.float32
    assert x.shape == out.shape

    M, N = x.shape
    stride_row = x.stride(0)

    BLOCK_SIZE = triton.next_power_of_2(N)

    grid = (M,)

    layernorm_kernel[grid](
        x, gamma, beta, out,
        M, N,
        stride_row,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )