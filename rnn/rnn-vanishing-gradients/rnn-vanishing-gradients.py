import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    s = np.linalg.norm(W_hh, ord=2)
    return [s**t for t in range(T)]