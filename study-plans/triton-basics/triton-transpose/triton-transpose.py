import torch
import triton
import triton.language as tl


@triton.jit
def transpose_kernel(
    a_ptr, out_ptr,
    M, N,
    stride_am, stride_an,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask = (rm[:, None] < M) & (rn[None, :] < N)

    # Load tile of A in A's native coordinate system: rows -> M, cols -> N.
    a_ptrs = a_ptr + rm[:, None] * stride_am + rn[None, :] * stride_an
    tile = tl.load(a_ptrs, mask=mask, other=0.0)

    # Store into out at swapped coordinates: out[j, i] = A[i, j].
    # rm indexes columns of out, rn indexes rows of out -> swap the strides.
    out_ptrs = out_ptr + rm[:, None] * stride_on + rn[None, :] * stride_om
    tl.store(out_ptrs, tile, mask=mask)


def solve(a: torch.Tensor, out: torch.Tensor) -> None:
   
    assert a.is_cuda and out.is_cuda
    assert a.dtype == torch.float32

    M, N = a.shape
    assert out.shape == (N, M)

    BLOCK_M, BLOCK_N = 32, 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    transpose_kernel[grid](
        a, out,
        M, N,
        a.stride(0), a.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
    )