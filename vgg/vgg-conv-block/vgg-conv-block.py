import numpy as np

def vgg_conv_block(x: np.ndarray, weights: list, biases: list) -> np.ndarray:
    out = x
    for W, b in zip(weights, biases):
        out = np.maximum(0, out @ W + b)
    return out