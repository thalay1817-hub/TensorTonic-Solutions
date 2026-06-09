import numpy as np

def add_position_embedding(patches: np.ndarray, num_patches: int, embed_dim: int, pos_embed: np.ndarray = None) -> np.ndarray:
    if pos_embed is None:
        pos_embed = np.random.randn(1, num_patches, embed_dim) * 0.02
    return patches + np.array(pos_embed)