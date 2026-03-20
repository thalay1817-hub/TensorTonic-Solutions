import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Error
    e = y_true - y_pred
    abs_e = np.abs(e)

    # Apply piecewise formula
    loss = np.where(
        abs_e <= delta,
        0.5 * (e ** 2),                  # L2 region
        delta * (abs_e - 0.5 * delta)    # L1 region
    )

    # Return mean loss
    return np.mean(loss)