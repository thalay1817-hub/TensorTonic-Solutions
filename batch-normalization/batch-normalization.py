import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    x = np.array(x, dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta = np.array(beta, dtype=float)

    # Case 1: 2D input (N, D)
    if x.ndim == 2:
        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + eps)
        y = gamma * x_hat + beta

    # Case 2: 4D input (N, C, H, W)
    elif x.ndim == 4:
        mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
        var = np.var(x, axis=(0, 2, 3), keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + eps)

        # reshape gamma and beta for broadcasting
        gamma = gamma.reshape(1, -1, 1, 1)
        beta = beta.reshape(1, -1, 1, 1)

        y = gamma * x_hat + beta

    else:
        raise ValueError("Input must be 2D or 4D")

    return y