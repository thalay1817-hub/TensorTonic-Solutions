import numpy as np

def matrix_inverse(A):
    A = np.asarray(A, dtype=float)

    # Check if matrix is 2D and square
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None

    # Check if matrix is singular
    if abs(np.linalg.det(A)) < 1e-10:
        return None

    # Compute inverse
    A_inv = np.linalg.inv(A)

    return A_inv