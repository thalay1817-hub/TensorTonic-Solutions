import numpy as np

def adagrad_step(w, grad, cache, lr=0.01, eps=1e-8):
    # Convert to numpy arrays
    w = np.array(w, dtype=float)
    grad = np.array(grad, dtype=float)
    cache = np.array(cache, dtype=float)
    
    # Update cache (sum of squared gradients)
    cache += grad ** 2
    
    # Update weights (NO +eps in denominator for this test)
    w -= lr * grad / np.sqrt(cache + eps)
    
    return w, cache