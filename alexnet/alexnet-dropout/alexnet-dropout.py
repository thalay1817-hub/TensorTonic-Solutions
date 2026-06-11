import numpy as np

def dropout(x: np.ndarray, p: float = 0.5, training: bool = True, mask: np.ndarray = None) -> np.ndarray:
    if not training:
        return x
    if mask is None:
        mask = np.random.binomial(1, 1 - p, size=x.shape)
    return x * mask / (1 - p)