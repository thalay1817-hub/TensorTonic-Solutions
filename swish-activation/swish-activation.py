import numpy as np

def swish(x):
    # Convert input to numpy array
    x = np.asarray(x, dtype=float)
    
    # Clip values for numerical stability (avoid overflow in exp)
    x_clipped = np.clip(x, -500, 500)
    
    # Sigmoid function
    sigmoid = 1 / (1 + np.exp(-x_clipped))
    
    # Swish activation
    result = x * sigmoid
    
    return result


# Example usage
print(np.round(swish([0, 1, -1, 3]), 3))
print(swish(0.0))  # scalar input
print(np.round(swish([[1, -1], [2, -2]]), 3))