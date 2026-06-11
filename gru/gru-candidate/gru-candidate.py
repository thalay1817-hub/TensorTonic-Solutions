import numpy as np

def candidate_hidden(h_prev: np.ndarray, x_t: np.ndarray, r_t: np.ndarray,
                     W_h: np.ndarray, b_h: np.ndarray) -> np.ndarray:
    combined = np.concatenate([r_t * h_prev, x_t], axis=-1)
    return np.tanh(combined @ W_h.T + b_h)