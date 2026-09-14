import torch
import triton
import triton.language as tl


@triton.jit
def qk_matmul_kernel(
    q_ptr, k_ptr, s_ptr,
    M, N, D,
    stride_qm, stride_qd,
    stride_kn, stride_kd,
    stride_sm, stride_sn,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < M
    mask_n = offs_n < N
    mask_d = offs_d < D

    q_ptrs = q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    k_ptrs = k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale

    s_ptrs = s_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    tl.store(s_ptrs, scores, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def softmax_kernel(
    s_ptr, out_ptr,
    M, N,
    stride_m,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N

    row_ptr = s_ptr + row * stride_m
    x = tl.load(row_ptr + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x_shifted = x - row_max
    numerator = tl.exp(x_shifted)
    denom = tl.sum(numerator, axis=0)
    softmax = numerator / denom

    out_row_ptr = out_ptr + row * stride_m
    tl.store(out_row_ptr + col_offsets, softmax, mask=mask)


@triton.jit
def av_matmul_kernel(
    a_ptr, v_ptr, o_ptr,
    M, N, D,
    stride_am, stride_an,
    stride_vn, stride_vd,
    stride_om, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < M
    mask_d = offs_d < D

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        v_ptrs = v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(a, v, input_precision="ieee")

    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor):
    M, D = Q.shape
    N, _ = K.shape

    scale = 1.0 / (D ** 0.5)

    
    S = torch.empty((M, N), device=Q.device, dtype=torch.float32)
    A = torch.empty((M, N), device=Q.device, dtype=torch.float32)

    BLOCK_M = max(16, min(64, triton.next_power_of_2(M)))
    BLOCK_N = max(16, min(64, triton.next_power_of_2(N)))
    BLOCK_D = max(16, triton.next_power_of_2(D))

    
    grid1 = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    qk_matmul_kernel[grid1](
        Q, K, S,
        M, N, D,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        S.stride(0), S.stride(1),
        scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

   
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid2 = (M,)
    softmax_kernel[grid2](
        S, A,
        M, N,
        S.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

  
    grid3 = (triton.cdiv(M, BLOCK_M),)
    av_matmul_kernel[grid3](
        A, V, O,
        M, N, D,
        A.stride(0), A.stride(1),
        V.stride(0), V.stride(1),
        O.stride(0), O.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    return O