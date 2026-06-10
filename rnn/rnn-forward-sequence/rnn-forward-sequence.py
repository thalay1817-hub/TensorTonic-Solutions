import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    h = h_0
    hidden_states = []
    for t in range(X.shape[1]):
        h = np.tanh(X[:, t, :] @ W_xh.T + h @ W_hh.T + b_h)
        hidden_states.append(h)
    return np.stack(hidden_states, axis=1), h