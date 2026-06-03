import numpy as np

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    x     = np.array(x,     dtype=float)
    conv1 = np.array(conv1, dtype=float)
    W1_b1 = np.array(W1_b1, dtype=float)
    W2_b1 = np.array(W2_b1, dtype=float)
    W1_b2 = np.array(W1_b2, dtype=float)
    W2_b2 = np.array(W2_b2, dtype=float)
    Ws_b2 = np.array(Ws_b2, dtype=float)
    fc    = np.array(fc,    dtype=float)

    relu = lambda z: np.maximum(0, z)

    # Initial conv + ReLU
    out = relu(x @ conv1)

    # Block 1: identity shortcut (same dims)
    shortcut = out
    h = relu(out @ W1_b1)
    h = h @ W2_b1
    out = relu(h + shortcut)

    # Block 2: projection shortcut (dim change via Ws_b2)
    shortcut = out @ Ws_b2
    h = relu(out @ W1_b2)
    h = h @ W2_b2
    out = relu(h + shortcut)

    # FC layer (no activation — raw logits)
    return out @ fc