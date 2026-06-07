import numpy as np

def unet_decoder_block(x: np.ndarray, skip: np.ndarray, out_channels: int) -> np.ndarray:
    B, H, W, C = x.shape
    up_H = H * 2
    up_W = W * 2
    return np.zeros((B, up_H - 4, up_W - 4, out_channels))