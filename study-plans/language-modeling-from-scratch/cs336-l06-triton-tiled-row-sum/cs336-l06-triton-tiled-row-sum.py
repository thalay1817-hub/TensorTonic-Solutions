import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_kernel(
    x_ptr,
    out_ptr,
    n_cols,
    row_stride,
    col_stride,
    BLOCK_SIZE: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    row_base_ptr = x_ptr + row * row_stride

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for tile_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        ptrs = row_base_ptr + col_offsets * col_stride

        if IS_BF16:
            # Read raw bf16 bits as int16, widen manually to fp32
            # (avoids native .bf16 PTX which needs sm_80+).
            raw_i16 = tl.load(ptrs, mask=mask, other=0)
            raw_u16 = raw_i16.to(tl.uint16, bitcast=True)
            bits32 = raw_u16.to(tl.uint32) << 16
            vals = bits32.to(tl.float32, bitcast=True)
            vals = tl.where(mask, vals, 0.0)
        else:
            vals = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

        acc += vals

    row_sum = tl.sum(acc, axis=0)
    tl.store(out_ptr + row, row_sum)


def solve(x: torch.Tensor, out: torch.Tensor, tile_size: int):
  
    assert x.is_cuda and out.is_cuda, "x and out must be CUDA tensors"

    if x.dim() != 2:
        raise ValueError("x must be a rank-2 tensor")

    n_rows, n_cols = x.shape

    if out.dim() != 1 or out.shape[0] != n_rows:
        raise ValueError("out must be a contiguous float32 vector of length equal to row count")
    if out.dtype != torch.float32:
        raise ValueError("out must be float32")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")

    if not (tile_size >= 32 and tile_size <= 1024 and (tile_size & (tile_size - 1)) == 0):
        raise ValueError("tile_size must be a power of two from 32 through 1024")

    # Zero rows: return safely, nothing to write.
    if n_rows == 0:
        return

    # Zero-width rows: every row sum is trivially zero.
    if n_cols == 0:
        out.zero_()
        return

    row_stride = x.stride(0)
    col_stride = x.stride(1)

    is_bf16 = x.dtype == torch.bfloat16
    x_view = x.view(torch.int16) if is_bf16 else x

    grid = (n_rows,)
    row_sum_kernel[grid](
        x_view,
        out,
        n_cols,
        row_stride,
        col_stride,
        BLOCK_SIZE=tile_size,
        IS_BF16=is_bf16,
    )