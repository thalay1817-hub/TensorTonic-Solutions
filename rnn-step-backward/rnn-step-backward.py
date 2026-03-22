import numpy as np

def rnn_step_backward(dh, cache):
    x_t, h_prev, h_t, W, U, b = cache
    
    # Convert to numpy arrays
    dh = np.array(dh, dtype=float)
    x_t = np.array(x_t, dtype=float)
    h_prev = np.array(h_prev, dtype=float)
    h_t = np.array(h_t, dtype=float)
    W = np.array(W, dtype=float)
    U = np.array(U, dtype=float)
    
    # Step 1: gradient through tanh
    dz = dh * (1 - h_t**2)   # element-wise
    
    # Step 2: gradients
    dx = W.T @ dz            # (D,)
    dh_prev = U.T @ dz       # (H,)
    
    dW = np.outer(dz, x_t)   # (H, D)
    dU = np.outer(dz, h_prev) # (H, H)
    db = dz                  # (H,)
    
    return (
        dx.tolist(),
        dh_prev.tolist(),
        dW.tolist(),
        dU.tolist(),
        db.tolist()
    )