import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    X = np.array(X, dtype=float)
    x_min = np.min(X, axis=axis, keepdims=True)
    x_max = np.max(X, axis=axis, keepdims=True)
    return (X - x_min) / np.maximum(x_max - x_min, eps)

   