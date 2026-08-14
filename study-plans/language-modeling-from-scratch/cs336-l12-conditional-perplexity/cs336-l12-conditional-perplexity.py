import torch

def conditional_perplexity(logits, target_ids, response_mask):
    L = logits.detach().clone()
    ids = target_ids.detach().clone()
    mask = response_mask.detach().clone()

    selected_token_count = int(torch.sum(mask).item())

    if selected_token_count == 0:
        return {
            "negative_log_likelihood": 0.0,
            "perplexity": 1.0,
            "selected_token_count": 0,
        }

    log_probs = torch.log_softmax(L.to(torch.float64), dim=-1)
    target_log_probs = torch.gather(log_probs, dim=-1, index=ids.unsqueeze(-1)).squeeze(-1)

    selected = target_log_probs[mask]
    nll = -torch.mean(selected)

    negative_log_likelihood = float(nll.item())
    perplexity = float(torch.exp(nll).item())

    return {
        "negative_log_likelihood": negative_log_likelihood,
        "perplexity": perplexity,
        "selected_token_count": selected_token_count,
    }