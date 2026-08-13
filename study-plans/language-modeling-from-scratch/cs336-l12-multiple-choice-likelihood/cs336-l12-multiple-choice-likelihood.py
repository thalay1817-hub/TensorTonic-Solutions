import torch

def multiple_choice_likelihood(
    logits,
    token_ids,
    continuation_mask,
):
    log_probs = torch.log_softmax(logits, dim=-1)

    target_log_probs = torch.gather(
        log_probs, dim=-1, index=token_ids.unsqueeze(-1)
    ).squeeze(-1)

    masked_log_probs = torch.where(
        continuation_mask,
        target_log_probs,
        torch.zeros_like(target_log_probs),
    )

    choice_scores = masked_log_probs.sum(dim=-1)

    predicted_indices = torch.argmax(choice_scores, dim=-1).to(torch.int64)

    return {
        "choice_scores": choice_scores,
        "predicted_indices": predicted_indices,
    }