import numpy as np

def pearson_correlation(X):
    X = np.array(X, dtype=float)
    
    # Validate input
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    
    N, D = X.shape
    
    # Step 1: Covariance matrix (D x D)
    X_centered = X - np.mean(X, axis=0)
    cov = (X_centered.T @ X_centered) / (N - 1)
    
    # Step 2: Standard deviations vector (D,)
    std = np.std(X, axis=0, ddof=1)
    
    # Step 3: Outer product of std vectors → denominator matrix
    denom = np.outer(std, std)
    
    # Step 4: Divide covariance by denominator
    with np.errstate(invalid='ignore'):
        R = cov / denom
    
    # Step 5: Zero-variance features → entire row & col = NaN (including diagonal)
    zero_var = (std == 0)
    R[zero_var, :] = np.nan
    R[:, zero_var] = np.nan

    # Step 6: Diagonal → 1.0 ONLY for non-zero variance features
    for i in range(D):
        if not zero_var[i]:
            R[i, i] = 1.0
    
    return R