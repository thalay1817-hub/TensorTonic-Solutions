import numpy as np

def he_initialization(W, fan_in):
    W = np.asarray(W, dtype=float)

    # compute limit
    limit = np.sqrt(6 / fan_in)

    # scale from [0,1] → [-limit, limit]
    W_scaled = W * 2 * limit - limit

    return W_scaled


# Examples
print(np.round(he_initialization([[0.5, 0.5]], 2), 4))

print(np.round(he_initialization([[0], [1]], 2), 4))