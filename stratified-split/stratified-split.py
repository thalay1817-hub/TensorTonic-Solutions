import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    X = np.array(X)
    y = np.array(y)

    train_idx = []
    test_idx  = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]

        # Shuffle within class
        if rng is not None:
            rng.shuffle(cls_idx)
        else:
            np.random.shuffle(cls_idx)

        # Calculate test count, ensure at least 1 stays in train
        n_test = round(len(cls_idx) * test_size)
        n_test = min(n_test, len(cls_idx) - 1)
        n_test = max(n_test, 0)

        test_idx.extend(cls_idx[:n_test])
        train_idx.extend(cls_idx[n_test:])

    # Sort to preserve original ordering ← THE FIX
    train_idx = np.sort(train_idx)
    test_idx  = np.sort(test_idx)

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]