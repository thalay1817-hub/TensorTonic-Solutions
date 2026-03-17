import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    Z1 = np.asarray(Z1, dtype=float)
    Z2 = np.asarray(Z2, dtype=float)
    
    N = Z1.shape[0]
    
    # Step 1: Similarity matrix
    S = np.dot(Z1, Z2.T) / temperature  # (N, N)
    
    # Step 2: Numerical stability (subtract row-wise max)
    S_max = np.max(S, axis=1, keepdims=True)
    S_stable = S - S_max
    
    # Step 3: Exponentiate
    exp_S = np.exp(S_stable)
    
    # Step 4: Compute probabilities
    denom = np.sum(exp_S, axis=1)          # sum over rows
    pos = np.diag(exp_S)                   # positive pairs (diagonal)
    
    # Step 5: InfoNCE loss
    loss = -np.mean(np.log(pos / denom))
    
    return float(loss)


# Example usage
print(info_nce_loss([[1,0],[0,1]], [[1,0],[0,1]], 0.1))  # ~0.0
print(info_nce_loss([[1,0],[0,1]], [[0,1],[1,0]], 0.1))  # high loss
print(round(info_nce_loss([[1,0],[0,1]], [[1,0],[0,1]], 1.0), 2))  # ~0.31