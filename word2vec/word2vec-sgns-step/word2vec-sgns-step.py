import torch
def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor, center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> tuple:
    W_in = W_in.clone()
    W_out = W_out.clone()
    
    vc = W_in[center_id].clone()
    uo = W_out[pos_id].clone()
    un = W_out[neg_ids].clone()
    
    sig_pos = torch.sigmoid(vc @ uo)
    sig_neg = torch.sigmoid(vc @ un.T)  # shape (k,)
    
    grad_uo = (sig_pos - 1) * vc
    grad_un = sig_neg.unsqueeze(1) * vc.unsqueeze(0)  # (k, D)
    grad_vc = (sig_pos - 1) * uo + (sig_neg.unsqueeze(1) * un).sum(dim=0)
    
    W_out[pos_id] -= lr * grad_uo
    for i, nid in enumerate(neg_ids):
        W_out[nid] -= lr * grad_un[i]
    W_in[center_id] -= lr * grad_vc
    
    return W_in, W_out