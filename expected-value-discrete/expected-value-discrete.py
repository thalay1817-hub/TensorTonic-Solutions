import numpy as np

def expected_value_discrete(x, p):
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)

    # Validate probability sum
    if not np.allclose(np.sum(p), 1):
        raise ValueError("Probabilities must sum to 1")

    # Compute expected value
    return np.sum(x * p)


# Example 1
x = [1, 2, 3]
p = [0.2, 0.5, 0.3]
print(expected_value_discrete(x, p))

# Example 2
x = [1, 2, 3, 4]
p = [0.25, 0.25, 0.25, 0.25]
print(expected_value_discrete(x, p))