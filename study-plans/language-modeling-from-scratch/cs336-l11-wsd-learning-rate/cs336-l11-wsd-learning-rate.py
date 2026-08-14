import math
import torch

def wsd_learning_rate(total_steps, warmup_steps, stable_steps, peak_learning_rate, final_ratio):
    T = total_steps
    w = warmup_steps
    s = stable_steps
    d = T - w - s

    rates = []

    # Warmup phase
    for t in range(w):
        eta = peak_learning_rate * (t + 1) / w
        rates.append(eta)

    # Stable phase
    for _ in range(s):
        rates.append(peak_learning_rate)

    # Decay phase
    r = final_ratio
    for j in range(d):
        if d == 1:
            p = 1.0
        else:
            p = j / (d - 1)
        eta = peak_learning_rate * (r + (1 - r) * (1 + math.cos(math.pi * p)) / 2)
        rates.append(eta)

    return torch.tensor(rates, dtype=torch.float64)
