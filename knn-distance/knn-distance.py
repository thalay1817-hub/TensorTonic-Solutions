import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.array(X_train, dtype=float)
    X_test  = np.array(X_test,  dtype=float)

    # Handle 1D → 2D reshape
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    n_train = X_train.shape[0]
    n_test  = X_test.shape[0]

    # Pairwise Euclidean distances via broadcasting → shape (n_test, n_train)
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))

    # k may exceed training set size → pad with -1
    effective_k = min(k, n_train)

    # argsort each row → closest first
    sorted_idx = np.argsort(dists, axis=1)[:, :effective_k]

    # Pad with -1 if k > n_train
    if effective_k < k:
        pad = np.full((n_test, k - effective_k), -1, dtype=int)
        sorted_idx = np.hstack([sorted_idx, pad])

    return sorted_idx.astype(int)