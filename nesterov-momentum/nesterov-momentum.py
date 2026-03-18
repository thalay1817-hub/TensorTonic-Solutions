import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    # Convert inputs to numpy arrays
    w = np.array(w, dtype=float)
    v = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)
    
    # Step 1: Look-ahead position (not explicitly needed since grad is already given at w_look)
    # w_look = w - momentum * v
    
    # Step 2: Update velocity
    v = momentum * v + lr * grad
    
    # Step 3: Update weights
    w = w - v
    
    return w.tolist(), v.tolist()


# Example usage
print(nesterov_momentum_step([1.0, -1.0], [0.0, 0.0], [0.5, -0.25], lr=0.1, momentum=0.9))
print(nesterov_momentum_step([1.0, 2.0], [0.5, -0.3], [0.1, 0.2], lr=0.1, momentum=0.9))
print(nesterov_momentum_step([2.0], [0.0], [0.0], lr=0.1, momentum=0.9))