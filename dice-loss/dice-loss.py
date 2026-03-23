import numpy as np

def dice_loss(p, y, eps=1e-8):
    # Convert to numpy arrays and flatten
    p = np.array(p, dtype=float).flatten()
    y = np.array(y, dtype=float).flatten()
    
    # Compute intersection and sums
    intersection = np.sum(p * y)
    sum_p = np.sum(p)
    sum_y = np.sum(y)
    
    # Dice coefficient
    dice = (2 * intersection + eps) / (sum_p + sum_y + eps)
    
    # Dice loss
    loss = 1 - dice
    
    return float(loss)