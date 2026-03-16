import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    
    # Convert inputs and force integer type
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    # Infer number of classes
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max()) + 1)

    K = num_classes

    # Compute indices for bincount (must be int)
    indices = (y_true * K + y_pred).astype(np.int64)

    counts = np.bincount(indices, minlength=K*K)
    C = counts.reshape(K, K)

    if normalize == 'none':
        return C

    C = C.astype(float)

    if normalize == 'true':   # row normalization
        denom = np.sum(C, axis=1, keepdims=True)
        denom[denom == 0] = 1
        C = C / denom

    elif normalize == 'pred': # column normalization
        denom = np.sum(C, axis=0, keepdims=True)
        denom[denom == 0] = 1
        C = C / denom

    elif normalize == 'all':  # normalize by total
        total = np.sum(C)
        if total != 0:
            C = C / total

    return C