import numpy as np

def normalize_3d(v):
    v = np.array(v, dtype=float)

    # compute norms
    norms = np.linalg.norm(v, axis=-1, keepdims=True)

    # avoid division by zero
    norms_safe = np.where(norms > 1e-10, norms, 1.0)

    normalized = v / norms_safe

    # keep zero vectors unchanged
    normalized = np.where(norms > 1e-10, normalized, 0.0)

    return normalized


# Example 1
v = [3, 4, 0]
print(normalize_3d(v))

# Example 2
v = [[0, 0, 0], [1, 2, 2]]
print(normalize_3d(v))