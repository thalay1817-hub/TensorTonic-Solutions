import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_backward_kernel(
    x_ptr, gamma_ptr, dy_ptr,
    dx_ptr, dgamma_ptr, dbeta_ptr,
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
    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    dy_row_ptr = dy_ptr + row * stride_m
    dy = tl.load(dy_row_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # Recompute row statistics
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + eps)

    x_hat = diff * rstd

    dy_norm = dy * gamma
    dy_norm = tl.where(mask, dy_norm, 0.0)

    c1 = tl.sum(dy_norm, axis=0) / N
    c2 = tl.sum(dy_norm * x_hat, axis=0) / N

    dx = rstd * (dy_norm - c1 - x_hat * c2)

    dx_row_ptr = dx_ptr + row * stride_m
    tl.store(dx_row_ptr + col_offsets, dx, mask=mask)

    # dgamma and dbeta: accumulate across rows via atomic add
    dgamma_contrib = dy * x_hat
    tl.atomic_add(dgamma_ptr + col_offsets, dgamma_contrib, mask=mask)
    tl.atomic_add(dbeta_ptr + col_offsets, dy, mask=mask)


def solve(
    x: torch.Tensor,
    gamma: torch.Tensor,
    dy: torch.Tensor,
    dx: torch.Tensor,
    dgamma: torch.Tensor,
    dbeta: torch.Tensor,
    eps: float,
):
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)

    dgamma.zero_()
    dbeta.zero_()

    grid = (M,)
    layernorm_backward_kernel[grid](
        x, gamma, dy,
        dx, dgamma, dbeta,
        M, N,
        x.stride(0),
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return dx, dgamma, dbeta