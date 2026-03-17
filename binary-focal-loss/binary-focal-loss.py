import math

def binary_focal_loss(predictions, targets, alpha=1.0, gamma=2.0):
    n = len(predictions)
    total_loss = 0.0
    
    for p, y in zip(predictions, targets):
        # Step 1: Compute p_t
        if y == 1:
            p_t = p
        else:
            p_t = 1 - p
        
        # Avoid log(0)
        p_t = max(min(p_t, 1 - 1e-15), 1e-15)
        
        # Step 2: Compute focal loss
        loss = -alpha * ((1 - p_t) ** gamma) * math.log(p_t)
        total_loss += loss
    
    # Step 3: Mean loss
    return total_loss / n


# Example usage
print(round(binary_focal_loss([0.9], [1], 1.0, 2.0), 6))
print(round(binary_focal_loss([0.1], [1], 1.0, 2.0), 4))