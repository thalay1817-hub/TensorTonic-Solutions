import numpy as np

def covariance_matrix(X):
    # Convert input to numpy array
    X = np.asarray(X, dtype=float)

    # Check dimensions
    if X.ndim != 2:
        return None

    N, D = X.shape

    # Need at least 2 samples
    if N < 2:
        return None

    # Step 1: Compute mean of each feature
    mu = np.mean(X, axis=0)

    # Step 2: Center the data
    X_centered = X - mu

    # Step 3: Compute covariance matrix
    cov_matrix = (X_centered.T @ X_centered) / (N - 1)

    return cov_matrix


# Example 1
X = [[1, 2], [2, 3], [3, 4]]
print(covariance_matrix(X))

# Example 2
X = [[1, 0], [0, 1]]
print(covariance_matrix(X))

# Example 3
X = [[1, 2, 3]]
print(covariance_matrix(X))