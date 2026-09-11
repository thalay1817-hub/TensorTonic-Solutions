import torch
import triton
import triton.language as tl


@triton.jit
def gemv_kernel(
    a_ptr, x_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rm < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n0 in range(0, N, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N)
        col_mask = rn < N

        a_ptrs = a_ptr + rm[:, None] * stride_am + rn[None, :] * stride_an
        a_mask = row_mask[:, None] & col_mask[None, :]
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        x_chunk = tl.load(x_ptr + rn, mask=col_mask, other=0.0)

        acc += tl.sum(a_tile * x_chunk[None, :], axis=1)

    out_ptrs = out_ptr + rm
    tl.store(out_ptrs, acc, mask=row_mask)


def solve(a: torch.Tensor, x: torch.Tensor, out: torch.Tensor) -> None:
   
    assert a.is_cuda and x.is_cuda and out.is_cuda
    assert a.dtype == torch.float32 and x.dtype == torch.float32

    M, N = a.shape
    assert x.shape == (N,)
    assert out.shape == (M,)

    BLOCK_M = 32
    BLOCK_N = 32

    grid = (triton.cdiv(M, BLOCK_M),)

    gemv_kernel[grid](
        a, x, out,
        M, N,
        a.stride(0), a.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )