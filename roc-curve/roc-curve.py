import numpy as np

def roc_curve(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # sort by descending score (stable for ties)
    order = np.lexsort((1 - y_true, -y_score))
    y_true = y_true[order]
    y_score = y_score[order]

    # cumulative true positives
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)

    P = np.sum(y_true)
    N = len(y_true) - P

    # find indices where score changes (unique thresholds)
    distinct = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct, len(y_score) - 1]

    # select TP and FP at each threshold
    tps = tps[threshold_idxs]
    fps = fps[threshold_idxs]

    thresholds = y_score[threshold_idxs]

    # compute TPR and FPR
    tpr = tps / P if P > 0 else np.zeros_like(tps)
    fpr = fps / N if N > 0 else np.zeros_like(fps)

    # prepend starting point
    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    thresholds = np.r_[np.inf, thresholds]

    return fpr, tpr, thresholds


# Example 1
y_true = [0, 1]
y_score = [0.1, 0.9]
print(roc_curve(y_true, y_score))

# Example 2
y_true = [1, 0, 1, 0]
y_score = [0.9, 0.7, 0.4, 0.2]
print(roc_curve(y_true, y_score))