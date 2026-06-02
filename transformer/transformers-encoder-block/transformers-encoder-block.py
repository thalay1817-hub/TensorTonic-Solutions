import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x, gamma, beta, eps=1e-6):
    mean = np.mean(x, axis=-1, keepdims=True)
    var  = np.var(x,  axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads):
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    def split_heads(x):
        return x.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    Q_h = split_heads(Q @ W_q)
    K_h = split_heads(K @ W_k)
    V_h = split_heads(V @ W_v)

    scores  = Q_h @ K_h.transpose(0, 1, 3, 2) / np.sqrt(d_k)
    weights = softmax(scores, axis=-1)
    concat  = (weights @ V_h).transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

    return concat @ W_o

def feed_forward(x, W1, b1, W2, b2):
    return np.maximum(0, x @ W1 + b1) @ W2 + b2

def encoder_block(x, W_q, W_k, W_v, W_o, W1, b1, W2, b2,
                  gamma1, beta1, gamma2, beta2, num_heads):
    x = layer_norm(x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads),
                   gamma1, beta1)
    x = layer_norm(x + feed_forward(x, W1, b1, W2, b2),
                   gamma2, beta2)
    return x