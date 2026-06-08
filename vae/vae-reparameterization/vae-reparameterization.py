import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    std = np.exp(0.5 * log_var)
    return mu + std * epsilon
