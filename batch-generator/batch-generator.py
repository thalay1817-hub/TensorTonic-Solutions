import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    X = np.asarray(X)
    y = np.asarray(y)

    n = len(X)

    # Create indices
    indices = np.arange(n)

    # Shuffle indices
    if rng is not None:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)

    # Reorder X and y using shuffled indices
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Generate batches
    for i in range(0, n, batch_size):
        end = i + batch_size

        # Drop last batch if required
        if drop_last and end > n:
            break

        yield X_shuffled[i:end], y_shuffled[i:end]


# Example
X = [0,1,2,3,4,5,6]
y = [0,1,2,3,4,5,6]

for xb, yb in batch_generator(X, y, batch_size=3, drop_last=False):
    print("X:", xb, "y:", yb)