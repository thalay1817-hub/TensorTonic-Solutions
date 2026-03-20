import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):
    ap_list = []

    for y_true, y_score in zip(y_true_list, y_score_list):
        y_true = np.array(y_true)
        y_score = np.array(y_score)

        # Total relevant items (IMPORTANT: from full list)
        R = np.sum(y_true)
        if R == 0:
            ap_list.append(0.0)
            continue

        # Sort by descending score
        indices = np.argsort(-y_score)
        y_true_sorted = y_true[indices]

        # Apply cutoff k only to ranked list
        if k is not None:
            y_true_sorted = y_true_sorted[:k]

        # Cumulative relevant count
        cum_relevant = np.cumsum(y_true_sorted)

        # Precision at each rank
        precision_at_k = cum_relevant / (np.arange(1, len(y_true_sorted) + 1))

        # Compute AP (normalize by total R, not truncated R)
        ap = np.sum(precision_at_k * y_true_sorted) / R
        ap_list.append(ap)

    mAP = np.mean(ap_list) if ap_list else 0.0
    return mAP, ap_list