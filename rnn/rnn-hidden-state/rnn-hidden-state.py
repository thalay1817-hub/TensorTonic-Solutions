import numpy as np

def init_hidden(batch_size: int, hidden_dim: int) -> np.ndarray:
    return np.zeros((batch_size, hidden_dim))