import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    ece = 0.0

    # assign bins
    bin_idx = np.floor(y_pred * n_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)  # handle p = 1.0

    for b in range(n_bins):
        mask = (bin_idx == b)
        count = np.sum(mask)

        if count > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_pred[mask])
            ece += (count / n) * abs(acc - conf)

    return float(ece)


# Examples
print(expected_calibration_error([1,0,1,0], [0.9,0.9,0.9,0.9], 5))
print(expected_calibration_error(
    [0,0,1,1,0,1,1,1],
    [0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9],
    2
))