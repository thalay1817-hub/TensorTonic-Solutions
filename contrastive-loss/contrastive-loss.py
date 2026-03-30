import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean"):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    y = np.array(y, dtype=float)

    # Ensure 2D
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]

    # Distances
    d = np.linalg.norm(a - b, axis=1)

    # Loss per sample
    loss = y * (d ** 2) + (1 - y) * np.maximum(0, margin - d) ** 2

    # Apply reduction
    if reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    else:  # "mean"
        return loss.mean()