import numpy as np

def pca_projection(X, k):
    # Convert to numpy array
    X = np.asarray(X, dtype=float)
    
    # Step 1: Center the data
    means = np.mean(X, axis=0)
    X_centered = X - means
    
    # Step 2: Covariance matrix
    n = X.shape[0]
    C = (X_centered.T @ X_centered) / (n - 1)
    
    # Step 3: Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # Step 4: Sort eigenvectors by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    W = eigenvectors[:, idx[:k]]
    
    # Step 5: Project data
    X_proj = X_centered @ W
    
    return X_proj


# Example 1
X = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0]]
k = 1
print(np.round(pca_projection(X, k), 2))


# Example 2
X = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
k = 1
print(np.round(pca_projection(X, k), 2))