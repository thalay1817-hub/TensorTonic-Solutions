import numpy as np

def angle_between_3d(v, w):
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)

    # Compute norms
    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)

    # Handle zero vectors
    if norm_v < 1e-10 or norm_w < 1e-10:
        return np.nan

    # Compute cosine of angle
    cos_theta = np.dot(v, w) / (norm_v * norm_w)

    # Clip for numerical stability
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Compute angle
    theta = np.arccos(cos_theta)

    return float(theta)


# Examples
print(angle_between_3d([1,0,0], [0,1,0]))
print(angle_between_3d([1,2,3], [2,4,6]))