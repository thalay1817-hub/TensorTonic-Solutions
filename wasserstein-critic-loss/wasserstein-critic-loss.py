import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    real_scores = np.array(real_scores, dtype=float)
    fake_scores = np.array(fake_scores, dtype=float)

    mean_real = np.mean(real_scores)
    mean_fake = np.mean(fake_scores)

    loss = mean_fake - mean_real

    return float(loss)