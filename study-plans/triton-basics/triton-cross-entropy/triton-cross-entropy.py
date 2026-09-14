import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr, target_ptr, loss_ptr,
    B, C,
    stride_b,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < C

    row_ptr = logits_ptr + row * stride_b
    logits = tl.load(row_ptr + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)

    row_max = tl.max(logits, axis=0)
    shifted = logits - row_max
    exp_shifted = tl.exp(shifted)
    sum_exp = tl.sum(exp_shifted, axis=0)
    lse = tl.log(sum_exp) + row_max

    target = tl.load(target_ptr + row)
    target_logit = tl.load(row_ptr + target).to(tl.float32)

    loss_i = lse - target_logit

    tl.atomic_add(loss_ptr, loss_i)


def solve(logits: torch.Tensor, target: torch.Tensor, loss_out: torch.Tensor):
    B, C = logits.shape
    BLOCK_SIZE = triton.next_power_of_2(C)

    loss_out.zero_()

    grid = (B,)
    cross_entropy_kernel[grid](
        logits, target, loss_out,
        B, C,
        logits.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    loss_out.div_(B)
    return loss_out