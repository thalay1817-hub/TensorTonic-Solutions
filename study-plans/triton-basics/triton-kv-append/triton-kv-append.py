import torch
import triton
import triton.language as tl


@triton.jit
def kv_cache_append_kernel(
    k_new_ptr, v_new_ptr,
    k_cache_ptr, v_cache_ptr,
    pos,
    D,
    BLOCK_SIZE: tl.constexpr,
):
    pid_d = tl.program_id(0)

    offsets = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < D

    k_val = tl.load(k_new_ptr + offsets, mask=mask, other=0.0)
    v_val = tl.load(v_new_ptr + offsets, mask=mask, other=0.0)

    row_offset = pos * D + offsets

    tl.store(k_cache_ptr + row_offset, k_val, mask=mask)
    tl.store(v_cache_ptr + row_offset, v_val, mask=mask)


def solve(
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos: int,
) -> None:
   
    assert k_new.is_cuda and v_new.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    D = k_new.shape[0]
    assert v_new.shape[0] == D
    assert k_cache.shape[1] == D
    assert v_cache.shape[1] == D

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(D, BLOCK_SIZE),)

    kv_cache_append_kernel[grid](
        k_new, v_new,
        k_cache, v_cache,
        pos,
        D,
        BLOCK_SIZE=BLOCK_SIZE,
    )