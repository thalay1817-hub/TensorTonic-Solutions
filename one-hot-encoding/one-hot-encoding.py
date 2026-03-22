import numpy as np

def one_hot(y, num_classes=None):
    y = np.array(y, dtype=int)
    
    # Determine number of classes
    if num_classes is None:
        num_classes = np.max(y) + 1
    
    # Validate labels
    if np.any(y < 0) or np.any(y >= num_classes):
        raise ValueError("Labels out of valid range [0, num_classes-1]")
    
    # Create zero matrix
    N = len(y)
    one_hot = np.zeros((N, num_classes), dtype=int)
    
    # Set correct positions to 1
    one_hot[np.arange(N), y] = 1
    
    return one_hot