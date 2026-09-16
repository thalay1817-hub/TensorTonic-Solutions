import torch
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    N, D,
    stride_xn, stride_xd,
    stride_cn, stride_cd,
    stride_on, stride_od,
    HALF_D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
  
    pid_m = tl.program_id(0)

    if pid_m >= N:
        return

    
    for start in range(0, HALF_D, BLOCK_SIZE):
        j = start + tl.arange(0, BLOCK_SIZE)
        mask = j < HALF_D

        
        even_off = pid_m * stride_xn + (2 * j) * stride_xd
        odd_off = pid_m * stride_xn + (2 * j + 1) * stride_xd

        x_even = tl.load(x_ptr + even_off, mask=mask, other=0.0)
        x_odd = tl.load(x_ptr + odd_off, mask=mask, other=0.0)

        cos_off = pid_m * stride_cn + j * stride_cd
        sin_off = pid_m * stride_cn + j * stride_cd  

        c = tl.load(cos_ptr + cos_off, mask=mask, other=0.0)
        s = tl.load(sin_ptr + sin_off, mask=mask, other=0.0)

        out_even = x_even * c - x_odd * s
        out_odd = x_even * s + x_odd * c

        out_even_off = pid_m * stride_on + (2 * j) * stride_od
        out_odd_off = pid_m * stride_on + (2 * j + 1) * stride_od

        tl.store(out_ptr + out_even_off, out_even, mask=mask)
        tl.store(out_ptr + out_odd_off, out_odd, mask=mask)


def solve(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, out: torch.Tensor) -> None:
   
    assert x.is_cuda and cos.is_cuda and sin.is_cuda and out.is_cuda
    N, D = x.shape
    HALF_D = D // 2
    assert cos.shape == (N, HALF_D)
    assert sin.shape == (N, HALF_D)
    assert out.shape == (N, D)

    BLOCK_SIZE = triton.next_power_of_2(max(HALF_D, 1))
    BLOCK_SIZE = min(BLOCK_SIZE, 256)
    if BLOCK_SIZE == 0:
        BLOCK_SIZE = 1

    grid = (N,)

    rope_kernel[grid](
        x, cos, sin, out,
        N, D,
        x.stride(0), x.stride(1),
        cos.stride(0), cos.stride(1),
        out.stride(0), out.stride(1),
        HALF_D=HALF_D,
        BLOCK_SIZE=BLOCK_SIZE,
    )