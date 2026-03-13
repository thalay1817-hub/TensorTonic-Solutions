def warmup_decay_schedule(base_lr, warmup_steps, total_steps, current_step):
    
    # Warmup phase
    if current_step < warmup_steps:
        lr = base_lr * (current_step / warmup_steps)
    
    # Decay phase
    else:
        lr = base_lr * (total_steps - current_step) / (total_steps - warmup_steps)
    
    return lr


# Example 1
print(warmup_decay_schedule(0.1, 10, 100, 5))

# Example 2
print(warmup_decay_schedule(0.1, 10, 100, 55))