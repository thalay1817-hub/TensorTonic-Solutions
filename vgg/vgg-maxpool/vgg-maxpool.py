import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    N, H, W, C = x.shape
    return x.reshape(N, H//2, 2, W//2, 2, C).max(axis=2).max(axis=3)