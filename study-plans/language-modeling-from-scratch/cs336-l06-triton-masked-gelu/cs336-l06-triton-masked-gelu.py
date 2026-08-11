import torch
import triton
import triton.language as tl


@triton.jit
def gelu_tanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    if IS_BF16:
        # x_ptr/out_ptr point to an int16 view of the bf16 buffer.
        # bf16 is just the top 16 bits of fp32, so widen manually
        # instead of using Triton's native bf16 type (avoids sm_80+ PTX).
        raw_i16 = tl.load(x_ptr + offsets, mask=mask, other=0)
        raw_u16 = raw_i16.to(tl.uint16, bitcast=True)
        bits32 = raw_u16.to(tl.uint32) << 16
        x = bits32.to(tl.float32, bitcast=True)
    else:
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
    COEFF: tl.constexpr = 0.044715

    x_cubed = x * x * x
    inner = SQRT_2_OVER_PI * (x + COEFF * x_cubed)

    # tanh(z) = 1 - 2 / (exp(2z) + 1), via exp2 for stability/speed.
    LOG2E: tl.constexpr = 1.4426950408889634
    exp2z = tl.math.exp2(2.0 * inner * LOG2E)
    tanh_inner = 1.0 - 2.0 / (exp2z + 1.0)

    result = 0.5 * x * (1.0 + tanh_inner)

    if IS_BF16:
        rbits = result.to(tl.uint32, bitcast=True)
        # round-to-nearest-even when truncating fp32 -> bf16
        rounded = (rbits + 0x8000 + ((rbits >> 16) & 1)) >> 16
        out_u16 = rounded.to(tl.uint16)
        out_i16 = out_u16.to(tl.int16, bitcast=True)
        tl.store(out_ptr + offsets, out_i16, mask=mask)
    else:
        out_dtype = out_ptr.dtype.element_ty
        tl.store(out_ptr + offsets, result.to(out_dtype), mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor):
   
    assert x.shape == out.shape, "x and out must have the same shape"
    assert x.dtype == out.dtype, "x and out must have the same dtype"
    assert x.is_cuda and out.is_cuda, "x and out must be CUDA tensors"

    n_elements = x.numel()

    # Empty input: return safely, leave memory unchanged.
    if n_elements == 0:
        return

    is_bf16 = x.dtype == torch.bfloat16

    x_contig = x.contiguous()
    if is_bf16:
        # Reinterpret the raw bf16 bytes as int16 so the kernel never
        # touches Triton's native bfloat16 PTX path.
        x_flat = x_contig.view(-1).view(torch.int16)
        out_flat = out.view(-1).view(torch.int16)
    else:
        x_flat = x_contig.view(-1)
        out_flat = out.view(-1)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    gelu_tanh_kernel[grid](
        x_flat,
        out_flat,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_BF16=is_bf16,
    )