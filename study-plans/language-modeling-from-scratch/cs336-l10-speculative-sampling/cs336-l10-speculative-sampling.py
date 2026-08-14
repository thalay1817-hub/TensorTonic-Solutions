import torch

def speculative_sampling(draft_tokens, draft_probabilities, target_probabilities,
                                          acceptance_random_values, residual_random_value):
    device = draft_tokens.device

    dt = draft_tokens.detach().clone()
    dp = draft_probabilities.detach().clone()
    tp = target_probabilities.detach().clone()
    ar = acceptance_random_values.detach().clone()

    K = dt.shape[0]

    accepted_tokens = []
    accepted_count = 0
    rejected_at = None
    replacement_token = None

    for i in range(K):
        xi = int(dt[i].item())
        qi_xi = dp[i, xi].item()
        pi_xi = tp[i, xi].item()

        if qi_xi == 0.0 and pi_xi == 0.0:
            acceptance = 0.0
        elif qi_xi == 0.0:
            acceptance = 1.0
        else:
            ratio = pi_xi / qi_xi
            acceptance = ratio if ratio < 1.0 else 1.0

        r = ar[i].item()

        if r < acceptance:
            accepted_tokens.append(xi)
            accepted_count += 1
        else:
            rejected_at = i

            p_row = tp[i].to(torch.float64)
            q_row = dp[i].to(torch.float64)
            diff = p_row - q_row
            residual = torch.clamp(diff, min=0.0)
            total = torch.sum(residual)
            normalized = residual / total

            cumsum = torch.cumsum(normalized, dim=0)
            idx = torch.searchsorted(cumsum, torch.tensor(residual_random_value, dtype=torch.float64))
            replacement_token = int(idx.item())

            break

    if replacement_token is not None:
        accepted_tokens.append(replacement_token)

    tokens = torch.tensor(accepted_tokens, dtype=torch.int64, device=device)

    return {
        "tokens": tokens,
        "accepted_count": accepted_count,
        "rejected_at": rejected_at,
    }