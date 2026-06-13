import numpy as np

def ddpm_sample(x_T, betas, epsilon_preds, z_values=None):
    x = np.array(x_T, dtype=float)
    betas = np.array(betas, dtype=float)
    T = len(betas)
    
    alpha_bars = np.cumprod(1 - betas)
    
    z_idx = 0
    for i, t in enumerate(range(T, 0, -1)):
        beta_t = betas[t - 1]
        alpha_t = 1 - beta_t
        alpha_bar_t = alpha_bars[t - 1]
        
        epsilon_pred = np.array(epsilon_preds[i], dtype=float)
        
        # Posterior mean
        mean = (1 / np.sqrt(alpha_t)) * (x - (beta_t / np.sqrt(1 - alpha_bar_t)) * epsilon_pred)
        
        if t == 1:
            x = mean  # no noise at final step
        else:
            if z_values is not None and z_idx < len(z_values):
                z = np.array(z_values[z_idx], dtype=float)
                z_idx += 1
            else:
                z = np.random.randn(*x.shape)
            x = mean + np.sqrt(beta_t) * z
    
    return np.round(x, 4).tolist()