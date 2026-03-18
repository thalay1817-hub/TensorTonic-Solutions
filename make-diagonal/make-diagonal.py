import numpy as np

def make_diagonal(v):
    return np.diag(v)


# Example usage
print(make_diagonal([3, 5]))
print(make_diagonal([1.5]))
print(make_diagonal([0, 0, 2]))