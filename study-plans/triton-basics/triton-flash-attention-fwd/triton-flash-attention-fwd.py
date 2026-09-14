import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    M, N, D,
    stride_qn, stride_qd,
    stride_kn, stride_kd,
    stride_vn, stride_vd,
    stride_on, stride_od,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < M
    mask_d = offs_d < D

    # Load the Q row-block once; keep it in registers for the whole sweep
    q_ptrs = q_ptr + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # Running online-softmax state
    m_i = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        k_ptrs = k_ptr + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, -float('inf'))

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(p, v, input_precision="ieee")

        m_i = m_new

    acc = acc / l_i[:, None]

    o_ptrs = o_ptr + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, O: torch.Tensor):
    M, D = Q.shape   
    N, _ = K.shape   

  
    BLOCK_M = max(16, min(64, triton.next_power_of_2(M)))
    BLOCK_N = max(16, min(64, triton.next_power_of_2(N)))
    BLOCK_D = max(16, triton.next_power_of_2(D))

    scale = 1.0 / (D ** 0.5)

    grid = (triton.cdiv(M, BLOCK_M),)
    flash_attn_kernel[grid](
        Q, K, V, O,
        M, N, D,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        O.stride(0), O.stride(1),
        scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return O