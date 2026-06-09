import numpy as np

def discriminator(x, W):
    logits = np.dot(x, W)
    return np.round(1 / (1 + np.exp(-logits)), 4)