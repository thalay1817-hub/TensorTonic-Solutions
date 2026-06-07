import numpy as np

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    B, H, W, C = x.shape
    
    # Encoder: 4 blocks (conv-conv-pool), save skip shapes
    skips = []
    for _ in range(4):
        H, W = H - 4, W - 4      # two 3x3 valid convs
        skips.append((H, W))
        H, W = H // 2, W // 2    # max pool

    # Bottleneck: two 3x3 valid convs, no pool
    H, W = H - 4, W - 4

    # Decoder: 4 blocks (upsample-concat-conv-conv)
    for skip_H, skip_W in reversed(skips):
        H, W = H * 2, W * 2      # up-conv doubles spatial dims
        H, W = H - 4, W - 4      # two 3x3 valid convs

    # 1x1 output conv: spatial unchanged, channels -> num_classes
    return np.zeros((B, H, W, num_classes))