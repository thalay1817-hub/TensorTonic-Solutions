import numpy as np

def global_avg_pool(x):
    x = np.asarray(x, dtype=float)

    # Case 1: (C, H, W)
    if x.ndim == 3:
        return np.mean(x, axis=(1, 2))

    # Case 2: (N, C, H, W)
    elif x.ndim == 4:
        return np.mean(x, axis=(2, 3))

    else:
        raise ValueError("Input must have shape (C,H,W) or (N,C,H,W)")


# Example 1
x = np.ones((3, 2, 2))
print(global_avg_pool(x))


# Example 2
x = np.array([[[[1,2],[3,4]]]])
print(global_avg_pool(x))