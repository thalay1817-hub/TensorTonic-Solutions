import numpy as np

def _sigmoid(x):
    # numerically stable sigmoid
    return 1 / (1 + np.exp(-x))

def _as2d(a, feat):
    a = np.array(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    # Extract parameters
    Wz, Uz, bz = params["Wz"], params["Uz"], params["bz"]
    Wr, Ur, br = params["Wr"], params["Ur"], params["br"]
    Wh, Uh, bh = params["Wh"], params["Uh"], params["bh"]

    H = bz.shape[0]

    # Ensure 2D inputs
    x, x_was_1d = _as2d(x, Wz.shape[0])
    h_prev, h_was_1d = _as2d(h_prev, H)

    # Update gate
    z = _sigmoid(x @ Wz + h_prev @ Uz + bz)

    # Reset gate
    r = _sigmoid(x @ Wr + h_prev @ Ur + br)

    # Candidate hidden state
    h_tilde = np.tanh(x @ Wh + (r * h_prev) @ Uh + bh)

    # Final hidden state
    h = (1 - z) * h_prev + z * h_tilde

    # Return original shape
    if x_was_1d:
        return h.flatten()
    return h