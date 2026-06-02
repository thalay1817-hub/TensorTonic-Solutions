import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    positions = np.arange(seq_length).reshape(-1, 1)            # (seq_length, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))  # (d_model/2,)

    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(positions * div_term)   # even columns
    pe[:, 1::2] = np.cos(positions * div_term)   # odd columns

    return pe