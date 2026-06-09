import numpy as np

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                      Wq=None, Wk=None, Wv=None, Wo=None, W1=None, W2=None) -> np.ndarray:

    B, N, D = x.shape
    head_dim = D // num_heads
    hidden_dim = int(embed_dim * mlp_ratio)

    def layer_norm(z, eps=1e-6):
        mean = z.mean(axis=-1, keepdims=True)
        std = z.std(axis=-1, keepdims=True)
        return (z - mean) / (std + eps)

    def gelu(z):
        return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))

    def init(shape):
        return np.random.randn(*shape) * 0.02

    # Initialize weights if not provided
    Wq = np.array(Wq) if Wq is not None else init((D, D))
    Wk = np.array(Wk) if Wk is not None else init((D, D))
    Wv = np.array(Wv) if Wv is not None else init((D, D))
    Wo = np.array(Wo) if Wo is not None else init((D, D))
    W1 = np.array(W1) if W1 is not None else init((D, hidden_dim))
    W2 = np.array(W2) if W2 is not None else init((hidden_dim, D))

    # ── Step 1: Pre-LayerNorm ────────────────────────────────────────────
    x_norm = layer_norm(x)

    # ── Step 2: Multi-Head Self-Attention ────────────────────────────────
    Q = (x_norm @ Wq).reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = (x_norm @ Wk).reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = (x_norm @ Wv).reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

    # Scaled dot-product attention
    scale = np.sqrt(head_dim)
    scores = (Q @ K.transpose(0, 1, 3, 2)) / scale        # (B, heads, N, N)
    scores = scores - scores.max(axis=-1, keepdims=True)   # numerical stability
    attn = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)

    # Merge heads
    out = (attn @ V).transpose(0, 2, 1, 3).reshape(B, N, D)
    out = out @ Wo

    # ── Step 3: First residual ───────────────────────────────────────────
    x = x + out

    # ── Step 4: Pre-LayerNorm before MLP ────────────────────────────────
    x_norm = layer_norm(x)

    # ── Step 5: MLP ──────────────────────────────────────────────────────
    mlp_out = gelu(x_norm @ W1) @ W2

    # ── Step 6: Second residual ──────────────────────────────────────────
    return x + mlp_out