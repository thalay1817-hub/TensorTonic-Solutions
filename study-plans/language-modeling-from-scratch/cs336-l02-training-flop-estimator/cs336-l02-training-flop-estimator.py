def flop_estimator(matmuls, attention_flops=0):
    # Sum 2 * B * D * K over all matrix multiplications (does not mutate matmuls)
    matmul_flops = sum(2 * B * D * K for B, D, K in matmuls)

    forward_flops = matmul_flops + attention_flops
    backward_flops = 2 * forward_flops
    total_flops = forward_flops + backward_flops

    return {
        "forward_flops": forward_flops,
        "backward_flops": backward_flops,
        "total_flops": total_flops,
    }