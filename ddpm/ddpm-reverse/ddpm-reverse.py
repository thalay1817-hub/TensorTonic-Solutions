import numpy as np

def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    x_t = np.array(x_t, dtype=float)
    epsilon_pred = np.array(epsilon_pred, dtype=float)
    betas = np.array(betas, dtype=float)

    beta_t = betas[t - 1]
    alpha_t = 1 - beta_t
    alpha_bar_t = np.cumprod(1 - betas)[t - 1]

    # Posterior mean
    mean = (1 / np.sqrt(alpha_t)) * (x_t - (beta_t / np.sqrt(1 - alpha_bar_t)) * epsilon_pred)

    # Add noise only if t > 1
    if t == 1:
        return np.round(mean, 4).tolist()
    else:
        if z is None:
            z = np.random.randn(*x_t.shape)
        else:
            z = np.array(z, dtype=float)
        sigma_t = np.sqrt(beta_t)
        x_prev = mean + sigma_t * z
        return np.round(x_prev, 4).tolist()