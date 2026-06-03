import numpy as np

def conv_block(x, W1, W2, Ws):
    x  = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    Ws = np.array(Ws)

    def relu(z):
        return np.maximum(0, z)

    shortcut = x @ Ws        # projection shortcut
    out = relu(x @ W1)       # first layer + ReLU
    out = out @ W2           # second layer — NO ReLU here
    
    return relu(out + shortcut)  # add shortcut, then final ReLU