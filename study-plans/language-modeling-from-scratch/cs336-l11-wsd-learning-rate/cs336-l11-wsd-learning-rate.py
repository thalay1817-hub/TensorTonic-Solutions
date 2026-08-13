import math
import torch

def wsd_learning_rate(
    total_steps,
    warmup_steps,
    stable_steps,
    peak_learning_rate,
    final_ratio,
):
    decay_steps = total_steps - warmup_steps - stable_steps

    rates = []

    for t in range(warmup_steps):
        rates.append(peak_learning_rate * (t + 1) / warmup_steps)

    for _ in range(stable_steps):
        rates.append(peak_learning_rate)

    for j in range(decay_steps):
        if decay_steps == 1:
            pj = 1.0
        else:
            pj = j / (decay_steps - 1)
        cosine_term = (1 + math.cos(math.pi * pj)) / 2
        eta_j = peak_learning_rate * (final_ratio + (1 - final_ratio) * cosine_term)
        rates.append(eta_j)

    return torch.tensor(rates, dtype=torch.float64)