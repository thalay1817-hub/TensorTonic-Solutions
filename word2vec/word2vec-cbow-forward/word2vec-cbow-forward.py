import torch
import torch.nn.functional as F
def cbow_forward(context_ids: torch.Tensor, target_id: int, W_in: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    h = W_in[context_ids].mean(dim=0)
    logits = W_out @ h
    return -F.log_softmax(logits, dim=0)[target_id]
