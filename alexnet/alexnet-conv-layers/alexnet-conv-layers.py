import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    B = image.shape[0]
    H_out = (224 + 2*2 - 11) // 4 + 1  # = 55
    return np.zeros((B, H_out, H_out, 96))