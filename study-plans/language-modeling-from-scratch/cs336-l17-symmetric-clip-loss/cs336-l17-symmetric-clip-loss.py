import torch
import torch.nn.functional as F

def symmetric_clip_loss(image_embeddings, text_embeddings, temperature):
    dtype = image_embeddings.dtype
    device = image_embeddings.device

    n = image_embeddings.shape[0]

    # L2-normalize along feature dimension
    image_normalized = F.normalize(image_embeddings, dim=1)
    text_normalized = F.normalize(text_embeddings, dim=1)

    # Similarity matrix S_ij = (v_i . t_j) / temperature
    similarities = (image_normalized @ text_normalized.T) / temperature
    similarities = similarities.to(dtype)

    targets = torch.arange(n, dtype=torch.long, device=device)

    # image-to-text: rows of S are logits over text indices
    loss_i2t = F.cross_entropy(similarities, targets)
    # text-to-image: rows of S^T are logits over image indices
    loss_t2i = F.cross_entropy(similarities.T, targets)

    loss = (loss_i2t + loss_t2i) / 2

    # Top-1 accuracies
    if n > 0:
        image_to_text_preds = similarities.argmax(dim=1)
        text_to_image_preds = similarities.T.argmax(dim=1)

        image_to_text_accuracy = (image_to_text_preds == targets).to(dtype).mean()
        text_to_image_accuracy = (text_to_image_preds == targets).to(dtype).mean()
    else:
        image_to_text_accuracy = torch.zeros((), dtype=dtype, device=device)
        text_to_image_accuracy = torch.zeros((), dtype=dtype, device=device)

    return {
        "loss": loss.to(dtype),
        "similarities": similarities,
        "image_to_text_accuracy": image_to_text_accuracy.to(dtype),
        "text_to_image_accuracy": text_to_image_accuracy.to(dtype),
    }