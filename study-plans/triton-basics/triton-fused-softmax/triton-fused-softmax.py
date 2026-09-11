import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_row,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N

    row_ptr = x_ptr + row * stride_row
    x = tl.load(row_ptr + col_offsets, mask=mask, other=-float("inf"))

    # Numerically stable softmax: subtract row max before exponentiating.
    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max
    numerator = tl.exp(x_shifted)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    out_row_ptr = out_ptr + row * stride_row
    tl.store(out_row_ptr + col_offsets, result, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
 
    assert x.is_cuda and out.is_cuda
    assert x.dtype == torch.float32
    assert x.shape == out.shape

    M, N = x.shape
    stride_row = x.stride(0)

    # BLOCK_N must be a power of 2 and >= N to cover the whole row in one tile.
    BLOCK_N = triton.next_power_of_2(N)

    grid = (M,)

    softmax_kernel[grid](
        x,
        out,
        M,
        N,
        stride_row,
        BLOCK_N=BLOCK_N,
    )