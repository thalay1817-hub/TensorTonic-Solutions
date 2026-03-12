import math

def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    lr = min_lr + 0.5 * (base_lr - min_lr) * (
        1 + math.cos(math.pi * current_step / total_steps)
    )
    return lr


# Example 1
print(cosine_annealing_schedule(0.1, 0.0, 100, 0))

# Example 2
print(cosine_annealing_schedule(0.1, 0.0, 100, 50))