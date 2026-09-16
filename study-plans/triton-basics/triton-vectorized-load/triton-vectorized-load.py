import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    result = x + y

    tl.store(out_ptr + offsets, result, mask=mask)


def solve(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
   
    assert x.is_cuda and y.is_cuda and out.is_cuda
    N = x.shape[0]
    assert y.shape[0] == N
    assert out.shape[0] == N

    BLOCK_SIZE = 4096
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    vector_add_kernel[grid](
        x, y, out,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )