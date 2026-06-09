import numpy as np

class GAN:
    def __init__(self, G_W, D_W):
        self.G_W = np.array(G_W, dtype=float)
        self.D_W = np.array(D_W, dtype=float)

    def generate(self, z):
        z = np.array(z, dtype=float)
        output = np.tanh(np.dot(z, self.G_W))
        return np.round(output, 4).tolist()

    def discriminate(self, x):
        x = np.array(x, dtype=float)
        logits = np.dot(x, self.D_W)
        probs = 1 / (1 + np.exp(-logits))
        return np.round(probs, 4).tolist()

    def train_step(self, real_data, z):
        eps = 1e-8
        real_data = np.array(real_data, dtype=float)
        z = np.array(z, dtype=float)

        fake_data = np.array(self.generate(z))

        real_probs = np.clip(np.array(self.discriminate(real_data)), eps, 1 - eps)
        fake_probs = np.clip(np.array(self.discriminate(fake_data)), eps, 1 - eps)

        d_loss = float(-np.mean(np.log(real_probs) + np.log(1 - fake_probs)))
        g_loss = float(-np.mean(np.log(fake_probs)))

        return {"d_loss": round(d_loss, 4), "g_loss": round(g_loss, 4)}