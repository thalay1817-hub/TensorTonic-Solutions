import math

def label_smoothing_loss(predictions, target, epsilon):
    K = len(predictions)
    loss = 0.0
    
    for i in range(K):
        # Smoothed label
        if i == target:
            q = (1 - epsilon) + (epsilon / K)
        else:
            q = epsilon / K
        
        # Avoid log(0)
        p = max(predictions[i], 1e-15)
        
        # Cross-entropy contribution
        loss += -q * math.log(p)
    
    return loss


# Example usage
print(round(label_smoothing_loss([0.9, 0.05, 0.05], 0, 0.1), 4))
print(round(label_smoothing_loss([0.7, 0.3], 0, 0.2), 4))