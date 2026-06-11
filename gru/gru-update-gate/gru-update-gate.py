import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def update_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_z: np.ndarray, b_z: np.ndarray) -> np.ndarray:
    combined = np.concatenate([h_prev, x_t], axis=-1)
    return sigmoid(combined @ W_z.T + b_z)