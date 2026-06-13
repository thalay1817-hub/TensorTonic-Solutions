import numpy as np

def linear_beta_schedule(T, beta_1=0.0001, beta_T=0.02):
    betas = np.linspace(beta_1, beta_T, T)
    return np.round(betas, 6).tolist()

def cosine_alpha_bar_schedule(T, s=0.008):
    t = np.arange(1, T + 1)
    f_t = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    f_0 = np.cos((s) / (1 + s) * np.pi / 2) ** 2
    alpha_bars = f_t / f_0
    alpha_bars = np.clip(alpha_bars, 0.0001, 0.9999)
    return np.round(alpha_bars, 6).tolist()

def alpha_bar_to_betas(alpha_bars):
    alpha_bars = np.array(alpha_bars, dtype=float)
    # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
    # For t=1, alpha_bar_{t-1} = 1 (no noise yet)
    alpha_bar_prev = np.concatenate([[1.0], alpha_bars[:-1]])
    betas = 1 - alpha_bars / alpha_bar_prev
    betas = np.clip(betas, 0.0001, 0.9999)
    return np.round(betas, 6).tolist()