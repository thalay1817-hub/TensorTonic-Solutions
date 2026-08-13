import torch

def conditional_perplexity(
    logits,
    target_ids,
    response_mask,
):
    selected_token_count = int(response_mask.sum().item())

    if selected_token_count == 0:
        return {
            "negative_log_likelihood": 0.0,
            "perplexity": 1.0,
            "selected_token_count": 0,
        }

    log_probs = torch.log_softmax(logits, dim=-1)
    target_log_probs = torch.gather(
        log_probs, dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)

    selected_log_probs = target_log_probs[response_mask]
    nll = -selected_log_probs.sum() / selected_token_count
    ppl = torch.exp(nll)

    return {
        "negative_log_likelihood": float(nll.item()),
        "perplexity": float(ppl.item()),
        "selected_token_count": selected_token_count,
    }