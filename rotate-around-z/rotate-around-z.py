import numpy as np

def rotate_around_z(points, theta):
    points = np.array(points, dtype=float)

    # Handle single point
    is_1d = (points.ndim == 1)
    if is_1d:
        points = points.reshape(1, 3)

    # Extract coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Rotation
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    x_new = x * cos_t - y * sin_t
    y_new = x * sin_t + y * cos_t
    z_new = z

    # Stack into numpy array
    rotated = np.stack([x_new, y_new, z_new], axis=1)

    # Return same dimensionality as input
    if is_1d:
        return rotated.reshape(3,)   # still numpy array
    return rotated