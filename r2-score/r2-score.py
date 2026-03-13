import numpy as np

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Handle constant target case
    if np.all(y_true == y_true[0]):
        if np.allclose(y_true, y_pred):
            return 1.0
        else:
            return 0.0

    # Compute mean of true values
    y_mean = np.mean(y_true)

    # Sum of squared errors (SSE)
    sse = np.sum((y_true - y_pred) ** 2)

    # Total sum of squares (SST)
    sst = np.sum((y_true - y_mean) ** 2)

    # R² score
    r2 = 1 - (sse / sst)

    return float(r2)


# Example 1
print(r2_score([3,4,5], [2.9,4.1,5.0]))

# Example 2
print(r2_score([1,1,1], [1,1,1]))

# Example 3
print(r2_score([1,1,1], [0,2,1]))