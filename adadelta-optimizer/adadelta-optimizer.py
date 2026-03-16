import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):

    # Convert to numpy arrays
    w = np.asarray(w, dtype=float)
    grad = np.asarray(grad, dtype=float)
    E_grad_sq = np.asarray(E_grad_sq, dtype=float)
    E_update_sq = np.asarray(E_update_sq, dtype=float)

    # Step 1: update gradient running average
    E_grad_sq_new = rho * E_grad_sq + (1 - rho) * (grad ** 2)

    # Step 2: compute parameter update
    delta_w = - (np.sqrt(E_update_sq + eps) / np.sqrt(E_grad_sq_new + eps)) * grad

    # Step 3: update squared update running average
    E_update_sq_new = rho * E_update_sq + (1 - rho) * (delta_w ** 2)

    # Step 4: update parameters
    w_new = w + delta_w

    return w_new, E_grad_sq_new, E_update_sq_new