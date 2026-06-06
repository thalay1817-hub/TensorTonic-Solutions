import numpy as np

def maxpool_2x2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H//2, 2, W//2, 2, C).max(axis=(2, 4))

def vgg_features(x: np.ndarray, config: list, conv_weights: list, conv_biases: list) -> np.ndarray:
    out = x
    w_idx = 0
    for entry in config:
        if entry == 'M':
            out = maxpool_2x2(out)
        else:
            out = np.maximum(0, out @ conv_weights[w_idx] + conv_biases[w_idx])
            w_idx += 1
    return out