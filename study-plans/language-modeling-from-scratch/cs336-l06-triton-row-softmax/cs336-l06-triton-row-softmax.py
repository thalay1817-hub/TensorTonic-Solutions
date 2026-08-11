import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    row_stride,
    BLOCK_SIZE: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row_x_ptr = x_ptr + row * row_stride
    row_out_ptr = out_ptr + row * n_cols  # output is contiguous

    if IS_BF16:
        # Read raw bf16 bits as int16, widen manually to fp32 to avoid
        # native .bf16 PTX (requires sm_80+).
        raw_i16 = tl.load(row_x_ptr + col_offsets, mask=mask, other=0)
        raw_u16 = raw_i16.to(tl.uint16, bitcast=True)
        bits32 = raw_u16.to(tl.uint32) << 16
        x = bits32.to(tl.float32, bitcast=True)
        x = tl.where(mask, x, float("-inf"))
    else:
        x = tl.load(row_x_ptr + col_offsets, mask=mask, other=float("-inf")).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max

    LOG2E: tl.constexpr = 1.4426950408889634
    numerator = tl.math.exp2(x_shifted * LOG2E)
    numerator = tl.where(mask, numerator, 0.0)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    if IS_BF16:
        rbits = result.to(tl.uint32, bitcast=True)
        rounded = (rbits + 0x8000 + ((rbits >> 16) & 1)) >> 16
        out_u16 = rounded.to(tl.uint16)
        out_i16 = out_u16.to(tl.int16, bitcast=True)
        tl.store(row_out_ptr + col_offsets, out_i16, mask=mask)
    else:
        out_dtype = out_ptr.dtype.element_ty
        tl.store(row_out_ptr + col_offsets, result.to(out_dtype), mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor):
   
    assert x.is_cuda and out.is_cuda, "x and out must be CUDA tensors"
    assert x.shape == out.shape, "x and out must have the same shape"
    assert x.dtype == out.dtype, "x and out must have the same dtype"

    if x.dim() != 2:
        raise ValueError("x must be a rank-2 tensor")

    n_rows, n_cols = x.shape

    # Column stride must be 1 (contiguous columns); row stride may be padded.
    if x.numel() == 0 and n_cols == 0:
        row_stride = 0
    else:
        row_stride = x.stride(0)
        col_stride = x.stride(1) if n_cols > 0 else 1
        if n_cols > 0 and col_stride != 1:
            raise ValueError("x must have contiguous columns (stride 1)")

    if n_cols < 1 or n_cols > 65536:
        raise ValueError("row width must be between 1 and 65536")

    if not out.is_contiguous():
        raise ValueError("out must be contiguous")

    # Zero rows: return safely, nothing to write.
    if n_rows == 0:
        return

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    is_bf16 = x.dtype == torch.bfloat16

    if is_bf16:
        x_view = x.view(torch.int16)
        out_view = out.view(torch.int16)
    else:
        x_view = x
        out_view = out

    grid = (n_rows,)
    softmax_kernel[grid](
        x_view,
        out_view,
        n_cols,
        row_stride,
        BLOCK_SIZE=BLOCK_SIZE,
        IS_BF16=is_bf16,
    )