import numpy as np

def adam_step(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    # Convert to numpy arrays
    param = np.asarray(param, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)

    # Update first moment
    m = beta1 * m + (1 - beta1) * grad

    # Update second moment
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Parameter update
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)

    return param_new, m, v


# Example usage
param = np.array([1.0])
grad = np.array([0.0])
m = np.array([0.0])
v = np.array([0.0])

param_new, m_new, v_new = adam_step(param, grad, m, v, t=1)

print("Updated param:", param_new)
print("m:", m_new)
print("v:", v_new)