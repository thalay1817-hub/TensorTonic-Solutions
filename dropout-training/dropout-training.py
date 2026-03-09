import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.asarray(x, dtype=float)

    # Random generator
    if rng is None:
        rand = np.random.random(x.shape)
    else:
        rand = rng.random(x.shape)

    # Create dropout pattern
    keep_prob = 1 - p
    pattern = (rand < keep_prob).astype(float) / keep_prob

    # Apply dropout
    output = x * pattern

    return output, pattern


# Example 1
x = [1., 2., 3.]
print(dropout(x, p=0.0))

# Example 2
x = [2., 4.]
print(dropout(x, p=0.5))