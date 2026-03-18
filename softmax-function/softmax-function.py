import numpy as np

def softmax(x):
    x = np.array(x, dtype=float)
    
    if x.ndim == 1:
        # 1D case
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)
    
    elif x.ndim == 2:
        # 2D case (row-wise softmax)
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# Example usage
print(softmax([1, 2, 3]))
print(softmax([[1, 2, 3], [0, 0, 0]]))