import numpy as np

def detect_skew(train_dist, serving_dist, threshold=0.2, eps=1e-10):
    result = {}

    for feature in train_dist:
        p_train   = np.array(train_dist[feature],   dtype=float) + eps
        p_serving = np.array(serving_dist[feature], dtype=float) + eps

        # PSI = sum((serving - train) * ln(serving / train))
        psi = float(np.sum((p_serving - p_train) * np.log(p_serving / p_train)))

        result[feature] = {
            "psi":    psi,
            "skewed": bool(psi >= threshold)
        }

    return result