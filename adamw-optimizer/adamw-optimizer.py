import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    w    = np.array(w,    dtype=float)
    m    = np.array(m,    dtype=float)
    v    = np.array(v,    dtype=float)
    grad = np.array(grad, dtype=float)

    # Step 1: Update first moment
    new_m = beta1 * m + (1 - beta1) * grad

    # Step 2: Update second moment
    new_v = beta2 * v + (1 - beta2) * grad ** 2

    # Step 3: AdamW update (decoupled weight decay)
    new_w = w - lr * (weight_decay * w) - lr * (new_m / (np.sqrt(new_v) + eps))

    return (new_w, new_m, new_v)