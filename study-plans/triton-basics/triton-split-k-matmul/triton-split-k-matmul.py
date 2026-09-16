import torch
import triton
import triton.language as tl


@triton.jit
def split_k_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

   
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak)
    b_ptrs = b_ptr + ((k_start + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_cur = k_start
    while k_cur < k_end:
        k_remaining = k_end - k_cur
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc, input_precision="ieee")

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        k_cur += BLOCK_K

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.atomic_add(c_ptrs, acc, mask=c_mask)


def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> None:
   
    assert A.is_cuda and B.is_cuda and C.is_cuda
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert C.shape == (M, N)

    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32

   
    if K >= 4 * max(M, N, 1) and K >= 128:
        SPLIT_K = 8
    elif K >= 2 * max(M, N, 1) and K >= 64:
        SPLIT_K = 4
    else:
        SPLIT_K = 1
    SPLIT_K = min(SPLIT_K, max(1, triton.cdiv(K, BLOCK_K)))

  
    C.zero_()

    grid = (
        triton.cdiv(M, BLOCK_M),
        triton.cdiv(N, BLOCK_N),
        SPLIT_K,
    )

    split_k_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
    )