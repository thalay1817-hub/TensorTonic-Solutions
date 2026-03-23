import numpy as np

def zscore_standardize(X, axis=0, eps=1e-8):
    X = np.asarray(X, dtype=float)
    
    # Mean and std along given axis
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    
    # Avoid division by zero
    std = std + eps
    
    # Standardization
    Z = (X - mean) / std
    
    return Z