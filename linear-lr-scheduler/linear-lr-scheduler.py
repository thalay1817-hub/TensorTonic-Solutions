def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0):
    # Warmup phase: 0 → initial_lr
    if warmup_steps > 0 and step < warmup_steps:
        return step * initial_lr / warmup_steps
    
    # Decay phase: initial_lr → final_lr
    elif step <= total_steps:
        return final_lr + (initial_lr - final_lr) * (
            (total_steps - step) / max(total_steps - warmup_steps, 1)
        )
    
    # After training: clamp at final_lr
    else:
        return final_lr