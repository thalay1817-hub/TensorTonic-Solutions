import numpy as np

def impute_missing(X, strategy='mean'):
    X = np.asarray(X, dtype=float)

    # Handle 1D case
    if X.ndim == 1:
        mask = np.isnan(X)
        valid = X[~mask]

        if valid.size == 0:
            fill = 0
        else:
            fill = np.mean(valid) if strategy == 'mean' else np.median(valid)

        X[mask] = fill
        return X

    # Handle 2D case
    X_filled = X.copy()
    rows, cols = X.shape

    for j in range(cols):
        col = X[:, j]
        mask = np.isnan(col)
        valid = col[~mask]

        if valid.size == 0:
            fill = 0
        else:
            if strategy == 'mean':
                fill = np.mean(valid)
            elif strategy == 'median':
                fill = np.median(valid)
            else:
                raise ValueError("strategy must be 'mean' or 'median'")

        X_filled[mask, j] = fill

    return X_filled


# Example 1
X = [[1, np.nan], [3, 5]]
print(impute_missing(X, 'mean'))

# Example 2
X = [[np.nan, 2], [np.nan, 4]]
print(impute_missing(X, 'median'))

# Example 3
X = [1, np.nan, 3, np.nan, 5]
print(impute_missing(X, 'mean'))