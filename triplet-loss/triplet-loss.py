import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    # Convert to NumPy arrays
    anchor = np.array(anchor)
    positive = np.array(positive)
    negative = np.array(negative)
    
    # Handle single vector case → reshape to (1, D)
    if anchor.ndim == 1:
        anchor = anchor.reshape(1, -1)
        positive = positive.reshape(1, -1)
        negative = negative.reshape(1, -1)
    
    # Squared Euclidean distances
    d_ap = np.sum((anchor - positive) ** 2, axis=1)
    d_an = np.sum((anchor - negative) ** 2, axis=1)
    
    # Triplet loss
    loss = np.maximum(0, d_ap - d_an + margin)
    
    # Return scalar if single sample, else mean loss
    return float(loss.mean())