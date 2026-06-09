import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int, cls_token: np.ndarray = None) -> np.ndarray:
    B = patches.shape[0]
    if cls_token is None:
        cls_token = np.random.randn(1, 1, embed_dim) * 0.02
    cls_tokens = np.tile(np.array(cls_token), (B, 1, 1))
    return np.concatenate([cls_tokens, patches], axis=1)