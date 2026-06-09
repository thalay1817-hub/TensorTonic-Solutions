import numpy as np

def train_gan_step(real_data, fake_data, D_W):
    eps = 1e-8

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    real_probs = np.clip(sigmoid(np.dot(real_data, D_W)), eps, 1 - eps)
    fake_probs = np.clip(sigmoid(np.dot(fake_data, D_W)), eps, 1 - eps)

    d_loss = float(-np.mean(np.log(real_probs) + np.log(1 - fake_probs)))
    g_loss = float(-np.mean(np.log(fake_probs)))

    return {"d_loss": round(d_loss, 4), "g_loss": round(g_loss, 4)}