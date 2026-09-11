import torch
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_ptr = x_ptr + row * x_row_stride
    x = tl.load(row_ptr + col_offsets, mask=mask, other=-float("inf"))

    # Numerically stable log-sum-exp: subtract row max before exponentiating.
    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max
    exp_shifted = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_shifted, axis=0)

    result = row_max + tl.log(sum_exp)

    tl.store(out_ptr + row, result)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    logsumexp_kernel[grid](
        x, out, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )