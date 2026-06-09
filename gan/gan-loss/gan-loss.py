import numpy as np

def discriminator_loss(real_probs, fake_probs):
    eps = 1e-8
    real = np.clip(real_probs, eps, 1 - eps)
    fake = np.clip(fake_probs, eps, 1 - eps)
    return round(float(-np.mean(np.log(real) + np.log(1 - fake))), 4)

def generator_loss(fake_probs):
    eps = 1e-8
    fake = np.clip(fake_probs, eps, 1 - eps)
    return round(float(-np.mean(np.log(fake))), 4)