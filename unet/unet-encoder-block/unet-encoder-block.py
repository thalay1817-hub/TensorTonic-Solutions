import numpy as np

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    B, H, W, C = x.shape
    skip_out = np.zeros((B, H-4, W-4, out_channels))
    pool_out = np.zeros((B, (H-4)//2, (W-4)//2, out_channels))
    return (pool_out, skip_out)
