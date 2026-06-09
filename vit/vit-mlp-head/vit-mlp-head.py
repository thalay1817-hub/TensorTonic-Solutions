import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int, W_head: np.ndarray = None) -> np.ndarray:
    # Step 1: Extract [CLS] token → (B, D)
    cls = encoder_output[:, 0, :]

    # Step 2: LayerNorm over last dim
    mean = cls.mean(axis=-1, keepdims=True)
    std = cls.std(axis=-1, keepdims=True)
    cls_norm = (cls - mean) / (std + 1e-6)

    # Step 3: Linear projection → (B, num_classes)
    if W_head is None:
        W_head = np.random.randn(cls.shape[-1], num_classes) * 0.02

    return cls_norm @ np.array(W_head)