import numpy as np

def euclidean_distance(x, y):
    # Convert to numpy arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Check same shape
    if x.shape != y.shape:
        raise ValueError("Input vectors must have the same shape")
    
    # Compute Euclidean distance
    dist = np.sqrt(np.sum((x - y) ** 2))
    
    return float(dist)