import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    recon = float(np.sum((x - x_recon)**2, axis=1).mean())
    kl = float(-0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1).mean())
    return {"total": recon + kl, "recon": recon, "kl": kl}