import numpy as np

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    x      = np.array(x,      dtype=float)
    W1     = np.array(W1,     dtype=float)
    W2     = np.array(W2,     dtype=float)
    gamma1 = np.array(gamma1, dtype=float)
    beta1  = np.array(beta1,  dtype=float)
    gamma2 = np.array(gamma2, dtype=float)
    beta2  = np.array(beta2,  dtype=float)

    def relu(z):
        return np.maximum(0, z)

    def bn(inp, gamma, beta, eps=1e-5):
        mean  = inp.mean(axis=0)
        var   = inp.var(axis=0)
        x_hat = (inp - mean) / np.sqrt(var + eps)
        return gamma * x_hat + beta

    if mode == "post":
        out = bn(x @ W1, gamma1, beta1)
        out = relu(out)
        out = bn(out @ W2, gamma2, beta2)
        out = relu(out + x)
    else:  # pre
        out = relu(bn(x, gamma1, beta1))
        out = relu(bn(out @ W1, gamma2, beta2))
        out = out @ W2 + x

    return {"output": out.tolist(), "mode": mode}