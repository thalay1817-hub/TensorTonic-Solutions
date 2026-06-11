import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)