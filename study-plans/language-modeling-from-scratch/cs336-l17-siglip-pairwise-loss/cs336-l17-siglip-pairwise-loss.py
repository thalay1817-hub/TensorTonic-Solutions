import torch
import torch.nn.functional as F

def siglip_pairwise_loss(image_embeddings, text_embeddings, logit_scale, bias):
    dtype = image_embeddings.dtype
    device = image_embeddings.device

    n = image_embeddings.shape[0]

    # L2-normalize along feature dimension
    image_normalized = F.normalize(image_embeddings, dim=1)
    text_normalized = F.normalize(text_embeddings, dim=1)

    # logits_ij = a * (v_i . t_j) + b
    logits = logit_scale * (image_normalized @ text_normalized.T) + bias
    logits = logits.to(dtype)

    # Pair labels: -1 everywhere, 1 on the diagonal
    pair_labels = torch.full((n, n), -1.0, dtype=dtype, device=device)
    if n > 0:
        pair_labels.fill_diagonal_(1.0)

    # Numerically stable loss: -mean(logsigmoid(z * logit))
    if n == 0:
        loss = torch.zeros((), dtype=dtype, device=device)
    else:
        loss = -F.logsigmoid(pair_labels * logits).mean()

    return {
        "loss": loss.to(dtype),
        "logits": logits,
        "pair_labels": pair_labels,
    }