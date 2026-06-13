import numpy as np

def get_alpha_bar(betas):
    return np.cumprod(1 - np.array(betas, dtype=float))

def forward_diffusion(x_0, t, betas, epsilon=None):
    x_0 = np.array(x_0, dtype=float)
    betas = np.array(betas, dtype=float)
    
    alpha_bar = get_alpha_bar(betas)
    alpha_bar_t = alpha_bar[t - 1]
    
    if epsilon is None:
        epsilon = np.random.randn(*x_0.shape)
    else:
        epsilon = np.array(epsilon, dtype=float)
    
    x_t = np.sqrt(alpha_bar_t) * x_0 + np.sqrt(1 - alpha_bar_t) * epsilon
    x_t = np.round(x_t, 4)
    
    return x_t.tolist()  # ← return only x_t, not a tuple