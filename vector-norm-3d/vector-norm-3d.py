import numpy as np

def vector_norm_3d(v):
    v = np.asarray(v, dtype=float)

    # Single vector case
    if v.ndim == 1:
        return float(np.sqrt(np.sum(v**2)))

    # Batch of vectors case
    elif v.ndim == 2:
        return np.sqrt(np.sum(v**2, axis=1))


# Example 1
v = [3, 4, 12]
print(vector_norm_3d(v))

# Example 2
v = [[1, 0, 0], [0, 3, 4]]
print(vector_norm_3d(v))