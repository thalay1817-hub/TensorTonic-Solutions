import numpy as np

class VisionTransformer:
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 W_patch=None, cls_token=None, pos_embed=None,
                 encoder_weights=None, W_head=None):

        self.patch_size = patch_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.embed_dim = D = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        N = self.num_patches

        C = 3
        patch_dim = patch_size * patch_size * C
        hidden_dim = int(D * mlp_ratio)

        def r(*shape): return np.random.randn(*shape) * 0.02

        # ── Patch projection ────────────────────────────────────────────
        self.W_patch = np.array(W_patch) if W_patch is not None else r(patch_dim, D)

        # ── CLS token & position embedding ──────────────────────────────
        self.cls_token = np.array(cls_token) if cls_token is not None else r(1, 1, D)
        self.pos_embed = np.array(pos_embed) if pos_embed is not None else r(1, N + 1, D)

        # ── Encoder block weights (per layer) ───────────────────────────
        if encoder_weights is not None:
            self.encoder_weights = [
                {k: np.array(v) for k, v in layer.items()}
                for layer in encoder_weights
            ]
        else:
            self.encoder_weights = [
                {"Wq": r(D,D), "Wk": r(D,D), "Wv": r(D,D), "Wo": r(D,D),
                 "W1": r(D, hidden_dim), "W2": r(hidden_dim, D)}
                for _ in range(depth)
            ]

        # ── Classification head ──────────────────────────────────────────
        self.W_head = np.array(W_head) if W_head is not None else r(D, num_classes)

    # ── Helpers ─────────────────────────────────────────────────────────
    @staticmethod
    def _layer_norm(x, eps=1e-6):
        return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + eps)

    @staticmethod
    def _gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def _encoder_block(self, x, w):
        B, N, D = x.shape
        num_heads = self.num_heads
        head_dim = D // num_heads

        # MSA with Pre-LN
        xn = self._layer_norm(x)
        Q = (xn @ w["Wq"]).reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
        K = (xn @ w["Wk"]).reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
        V = (xn @ w["Wv"]).reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(head_dim)
        scores -= scores.max(axis=-1, keepdims=True)
        attn = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)

        out = (attn @ V).transpose(0, 2, 1, 3).reshape(B, N, D) @ w["Wo"]
        x = x + out

        # MLP with Pre-LN
        x = x + (self._gelu(self._layer_norm(x) @ w["W1"]) @ w["W2"])
        return x

    # ── Forward pass ────────────────────────────────────────────────────
    def forward(self, image: np.ndarray) -> np.ndarray:
        B, H, W, C = image.shape
        P = self.patch_size
        nh, nw = H // P, W // P

        # Step 1: Patch embedding
        patches = (image.reshape(B, nh, P, nw, P, C)
                       .transpose(0, 1, 3, 2, 4, 5)
                       .reshape(B, nh * nw, P * P * C))
        z = patches @ self.W_patch                             # (B, N, D)

        # Step 2: Prepend CLS
        cls = np.tile(self.cls_token, (B, 1, 1))
        z = np.concatenate([cls, z], axis=1)                   # (B, N+1, D)

        # Step 3: Add position embeddings
        z = z + self.pos_embed                                 # (B, N+1, D)

        # Step 4: Encoder blocks
        for w in self.encoder_weights:
            z = self._encoder_block(z, w)

        # Step 5: Classification head
        cls_out = self._layer_norm(z[:, 0, :])                 # (B, D)
        return cls_out @ self.W_head                           # (B, num_classes)