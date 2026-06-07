import numpy as np

def unet_output(features: np.ndarray, num_classes: int) -> np.ndarray:
    B, H, W, C = features.shape
    return np.zeros((B, H, W, num_classes))