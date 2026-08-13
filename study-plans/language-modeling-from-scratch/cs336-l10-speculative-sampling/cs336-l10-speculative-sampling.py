import torch

def speculative_sampling(
    draft_tokens,
    draft_probabilities,
    target_probabilities,
    acceptance_random_values,
    residual_random_value,
):
    device = draft_tokens.device
    K = draft_tokens.shape[0]

    accepted_tokens = []
    accepted_count = K
    rejected_at = None

    for i in range(K):
        xi = int(draft_tokens[i].item())
        qi = float(draft_probabilities[i, xi].item())
        pi = float(target_probabilities[i, xi].item())

        if qi == 0.0 and pi == 0.0:
            a = 0.0
        elif qi == 0.0:
            a = 1.0
        else:
            a = min(1.0, pi / qi)

        r_accept = float(acceptance_random_values[i].item())

        if r_accept < a:
            accepted_tokens.append(xi)
        else:
            rejected_at = i
            accepted_count = i

            p_row = target_probabilities[i]
            q_row = draft_probabilities[i]
            residual = torch.clamp(p_row - q_row, min=0)
            total = residual.sum()
            normalized = residual / total
            cdf = torch.cumsum(normalized, dim=0)

            r_val = torch.tensor(
                residual_random_value, dtype=cdf.dtype, device=cdf.device
            )
            idx = torch.searchsorted(cdf, r_val).item()

            accepted_tokens.append(int(idx))
            break

    tokens = torch.tensor(accepted_tokens, dtype=torch.int64, device=device)

    return {
        "tokens": tokens,
        "accepted_count": int(accepted_count),
        "rejected_at": rejected_at,
    }