import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    x  = np.array(x, dtype=float)
    W1 = np.array(W1, dtype=float)
    W2 = np.array(W2, dtype=float)
    W3 = np.array(W3, dtype=float)
    Ws = np.array(Ws, dtype=float)

    def relu(z):
        return np.maximum(0, z)

    shortcut = x @ Ws         # projection shortcut (always applied)
    out = relu(x @ W1)        # 1x1 reduce  — ReLU ✓
    out = relu(out @ W2)      # 3x3 process — ReLU ✓
    out = out @ W3            # 1x1 expand  — NO ReLU ✗

    return relu(out + shortcut)  # final ReLU after skip add ✓