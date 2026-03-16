import numpy as np

def leaky_relu(x, alpha=0.01):
    x = np.asarray(x)                 # Convert input to NumPy array
    return np.where(x >= 0, x, alpha * x)

# Example 1
x1 = [-2, -1, 0, 1, 2]
print(leaky_relu(x1, 0.1))

# Example 2
x2 = [-5, 5]
print(leaky_relu(x2, 0.01))