import numpy as np

class VanillaRNN:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        self.W_xh = np.random.randn(hidden_dim, input_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / (2 * hidden_dim))
        self.W_hy = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_h = np.zeros(hidden_dim)
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray, h_0: np.ndarray = None) -> tuple:
        N, T, _ = X.shape
        h = np.zeros((N, self.hidden_dim)) if h_0 is None else h_0
        hidden_states = []
        for t in range(T):
            h = np.tanh(X[:, t, :] @ self.W_xh.T + h @ self.W_hh.T + self.b_h)
            hidden_states.append(h)
        all_h = np.stack(hidden_states, axis=1)          # (batch, T, hidden_dim)
        y_seq = all_h @ self.W_hy.T + self.b_y           # (batch, T, output_dim)
        return y_seq, h