import torch
import triton
import triton.language as tl


@triton.jit
def online_softmax_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    x_row_ptr = x_ptr + row * stride_xm
    out_row_ptr = out_ptr + row * stride_om

    
    m = -float("inf")
    l = 0.0

    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x = tl.load(x_row_ptr + offs * stride_xn, mask=mask, other=-float("inf"))

        chunk_max = tl.max(x, axis=0)
        m_new = tl.maximum(m, chunk_max)

        alpha = tl.exp(m - m_new)
        l = l * alpha

        p = tl.exp(x - m_new)
        p = tl.where(mask, p, 0.0)
        l = l + tl.sum(p, axis=0)

        m = m_new

   
    for start in range(0, N, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        x = tl.load(x_row_ptr + offs * stride_xn, mask=mask, other=-float("inf"))
        p = tl.exp(x - m)
        result = p / l

        tl.store(out_row_ptr + offs * stride_on, result, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
   
    assert x.is_cuda and out.is_cuda
    M, N = x.shape
    assert out.shape == (M, N)

    BLOCK_SIZE = 1024
    grid = (M,)

    online_softmax_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )