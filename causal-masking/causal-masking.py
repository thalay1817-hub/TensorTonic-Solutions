import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    scores = np.array(scores, copy=True)  # do not modify input
    
    T = scores.shape[-1]
    
    # Create upper triangular mask (True where j > i)
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    
    # Apply mask using broadcasting
    scores[..., mask] = mask_value
    
    return scores