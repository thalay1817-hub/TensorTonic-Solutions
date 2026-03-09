import numpy as np

def apply_homogeneous_transform(T, points):
    T = np.asarray(T, dtype=float)
    points = np.asarray(points, dtype=float)

    single = False
    if points.ndim == 1:          # single point case
        points = points.reshape(1, 3)
        single = True

    # Convert to homogeneous coordinates (append 1)
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))

    # Apply transform
    transformed_h = (T @ points_h.T).T

    # Extract spatial coordinates
    transformed = transformed_h[:, :3]

    if single:
        return transformed.reshape(3,)
    return transformed


# Example 1
T = [[1,0,0,1],
     [0,1,0,2],
     [0,0,1,3],
     [0,0,0,1]]

points = [0,0,0]

print(apply_homogeneous_transform(T, points))


# Example 2
T = [[0,-1,0,1],
     [1,0,0,0],
     [0,0,1,0],
     [0,0,0,1]]

points = [[1,0,0], [0,1,0]]

print(apply_homogeneous_transform(T, points))