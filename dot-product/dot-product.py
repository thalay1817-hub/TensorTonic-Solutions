import numpy as np

def dot_product(x, y):
    # Convert inputs to numpy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Compute dot product
    return float(np.dot(x, y))


# Example 1
print(dot_product([1,2,3], [4,5,6]))

# Example 2
print(dot_product([1,0], [0,1]))

# Example 3
print(dot_product([-1,2], [3,-1]))