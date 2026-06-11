import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    B, H, W, C = x.shape
    squared = x ** 2
    denom = np.zeros_like(x)

    half = n // 2
    for i in range(C):
        j_start = max(0, i - half)
        j_end = min(C, i + half + 1)
        denom[..., i] = (k + alpha * squared[..., j_start:j_end].sum(axis=-1)) ** beta

    return x / denom