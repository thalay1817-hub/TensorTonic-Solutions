import numpy as np

def clip_gradients(g, max_norm):
    g = np.asarray(g, dtype=float)
    
    # compute L2 norm
    norm = np.linalg.norm(g)
    
    # edge cases
    if norm == 0 or max_norm <= 0:
        return g.copy()
    
    # no clipping needed
    if norm <= max_norm:
        return g.copy()
    
    # scale gradients
    scale = max_norm / norm
    return g * scale