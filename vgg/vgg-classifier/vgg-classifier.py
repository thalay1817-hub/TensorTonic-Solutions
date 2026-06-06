import numpy as np

def vgg_classifier(features: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                   W2: np.ndarray, b2: np.ndarray, W3: np.ndarray, b3: np.ndarray) -> np.ndarray:
    x = features.reshape(features.shape[0], -1)   # (B, H*W*C)
    x = np.maximum(0, x @ W1 + b1)               # FC1 + ReLU
    x = np.maximum(0, x @ W2 + b2)               # FC2 + ReLU
    return x @ W3 + b3                            # FC3, raw logits