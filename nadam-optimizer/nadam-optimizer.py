import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    
    # Convert inputs to numpy arrays
    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    # Step 1: Update first moment
    m_new = beta1 * m + (1 - beta1) * grad

    # Step 2: Update second moment
    v_new = beta2 * v + (1 - beta2) * (grad ** 2)

    # Step 3: Nesterov adjusted update
    nesterov_term = beta1 * m_new + (1 - beta1) * grad
    w_new = w - lr * nesterov_term / (np.sqrt(v_new) + eps)

    return (w_new.tolist(), m_new.tolist(), v_new.tolist())


# Example
print(nadam_step(
    w=[1.0, -1.0],
    m=[0.1, -0.1],
    v=[0.01, 0.01],
    grad=[0.2, -0.3],
    lr=0.002
))