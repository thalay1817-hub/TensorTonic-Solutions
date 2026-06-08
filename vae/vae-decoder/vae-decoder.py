import numpy as np

def vae_decoder(z: np.ndarray, W_dec: np.ndarray, b_dec: np.ndarray) -> np.ndarray:
    return np.dot(z, W_dec) + b_dec