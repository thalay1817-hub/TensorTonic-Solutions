import numpy as np

def kl_divergence(p, q, eps=1e-12):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)

    # Add epsilon to q for numerical stability
    q = q + eps

    # Only keep positions where p > 0
    mask = p > 0

    # Compute KL: sum p[i] * log(p[i] / q[i])
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))