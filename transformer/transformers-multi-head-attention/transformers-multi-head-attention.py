import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    batch, seq_len, d_model = Q.shape
    d_k = d_model // num_heads

    # Project: (batch, seq, d_model)
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    # Split into heads: (batch, num_heads, seq, d_k)
    def split_heads(x):
        return x.reshape(batch, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    Q_h = split_heads(Q_proj)
    K_h = split_heads(K_proj)
    V_h = split_heads(V_proj)

    # Scaled dot-product attention per head
    scores = Q_h @ K_h.transpose(0, 1, 3, 2) / np.sqrt(d_k)  # (batch, heads, seq_q, seq_k)
    weights = softmax(scores, axis=-1)
    head_out = weights @ V_h                                    # (batch, heads, seq, d_k)

    # Concatenate heads: (batch, seq, d_model)
    concat = head_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

    return concat @ W_o