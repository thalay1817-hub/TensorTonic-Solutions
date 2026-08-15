import torch

def kl_regularized_policy_objective(policy_log_probs, reference_log_probs, rewards, response_mask, beta):
    device = policy_log_probs.device

    batch_size = policy_log_probs.shape[0]

    if batch_size == 0:
        objective = torch.zeros((), dtype=torch.float32, device=device)
        penalized_rewards = torch.zeros((0,), dtype=torch.float32, device=device)
        sampled_divergences = torch.zeros((0,), dtype=torch.float32, device=device)
        return {
            "objective": objective,
            "penalized_rewards": penalized_rewards,
            "sampled_divergences": sampled_divergences,
        }

    diff = policy_log_probs.to(torch.float32) - reference_log_probs.to(torch.float32)
    masked_diff = torch.where(
        response_mask,
        diff,
        torch.zeros_like(diff),
    )

    sampled_divergences = masked_diff.sum(dim=1).to(torch.float32)

    rewards_f32 = rewards.to(torch.float32)
    penalized_rewards = (rewards_f32 - beta * sampled_divergences).to(torch.float32)

    objective = penalized_rewards.mean().to(torch.float32)

    return {
        "objective": objective,
        "penalized_rewards": penalized_rewards,
        "sampled_divergences": sampled_divergences,
    }