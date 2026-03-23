import numpy as np

def kfold_split(N, k, shuffle=False, rng=None):
    # Create indices
    indices = np.arange(N)
    
    # Shuffle if required
    if shuffle:
        if rng is None:
            rng = np.random.default_rng()
        rng.shuffle(indices)
    
    # Split into k folds
    folds = np.array_split(indices, k)
    
    result = []
    
    for i in range(k):
        # Validation indices (NumPy array)
        val_idx = folds[i]
        
        # Training indices (NumPy array)
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        result.append((train_idx, val_idx))  # keep as NumPy arrays
    
    return result