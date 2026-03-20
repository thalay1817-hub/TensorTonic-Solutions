import numpy as np

def lbfgs_direction(grad, s_list, y_list):
    grad = np.array(grad, dtype=float)
    s_list = [np.array(s, dtype=float) for s in s_list]
    y_list = [np.array(y, dtype=float) for y in y_list]

    m = len(s_list)
    if m == 0:
        return -grad

    # Step 1: compute rho values
    rho = []
    for s, y in zip(s_list, y_list):
        rho.append(1.0 / np.dot(y, s))

    # First loop (backward)
    q = grad.copy()
    alpha = [0.0] * m

    for i in reversed(range(m)):
        alpha[i] = rho[i] * np.dot(s_list[i], q)
        q = q - alpha[i] * y_list[i]

    # Scaling (gamma using most recent pair)
    s_last = s_list[-1]
    y_last = y_list[-1]
    gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)

    r = gamma * q

    # Second loop (forward)
    for i in range(m):
        beta = rho[i] * np.dot(y_list[i], r)
        r = r + s_list[i] * (alpha[i] - beta)

    # Return descent direction
    return -r