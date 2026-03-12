import numpy as np

def tanh(x):
    # Convert input to numpy array
    x = np.asarray(x, dtype=float)
    
    # Apply tanh formula
    result = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    return result


# Example 1
print(np.round(tanh([0, 1, -1, 3]), 4))

# Example 2
print(np.round(tanh(0.0), 4))

# Example 3
print(np.round(tanh([[0, 1], [-1, 2]]), 4))