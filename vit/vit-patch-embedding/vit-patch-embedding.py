import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int, W_proj: np.ndarray = None) -> np.ndarray:
    B, H, W, C = image.shape
    P = patch_size
    nh, nw = H // P, W // P

    # Reshape: (B, nh, P, nw, P, C) → (B, nh, nw, P, P, C) → (B, N, P²C)
    patches = image.reshape(B, nh, P, nw, P, C)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(B, nh * nw, P * P * C)

    if W_proj is None:
        W_proj = np.random.randn(P * P * C, embed_dim) * 0.02

    return patches @ np.array(W_proj)