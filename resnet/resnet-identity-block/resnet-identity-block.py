import numpy as np

def identity_block(x, W1, W2):
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    
    def relu(z):
        return np.maximum(0, z)
    
    identity = x
    h = relu(x @ W1.T)
    out = h @ W2.T
    return relu(out + identity)