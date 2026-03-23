import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    # Try converting input
    try:
        X = np.array(matrix, dtype=float)
    except:
        return None

    # Must be 2D
    if X.ndim != 2:
        return None

    # Axis must be valid
    if axis not in (0, 1, None):
        return None

    # Norm type check
    if norm_type not in ('l1', 'l2', 'max'):
        return None

    # Compute norm
    if norm_type == 'l2':
        norm = np.sqrt(np.sum(X ** 2, axis=axis, keepdims=True))
    elif norm_type == 'l1':
        norm = np.sum(np.abs(X), axis=axis, keepdims=True)
    else:  # 'max'
        norm = np.max(np.abs(X), axis=axis, keepdims=True)

    # Prevent division by zero
    norm[norm == 0] = 1

    # Normalize
    return X / norm