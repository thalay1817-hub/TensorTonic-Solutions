import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    
    # Reshape 1D to 2D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    # Vectorized pairwise Euclidean distances (n_test, n_train)
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    # Handle k > n_train by padding with -1
    if k <= n_train:
        indices = np.argsort(distances, axis=1)[:, :k]
    else:
        indices = np.argsort(distances, axis=1)
        pad_width = k - n_train
        padding = np.full((n_test, pad_width), -1, dtype=int)
        indices = np.hstack([indices, padding])
    
    return indices.astype(int)  # ← return numpy array, not list