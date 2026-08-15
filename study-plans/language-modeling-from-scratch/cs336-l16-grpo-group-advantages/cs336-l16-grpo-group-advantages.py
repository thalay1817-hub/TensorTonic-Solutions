import torch

def grpo_group_advantages(rewards, group_ids, epsilon):
    dtype = rewards.dtype
    device = rewards.device

    n = rewards.shape[0]

    advantages = torch.zeros(n, dtype=dtype, device=device)

    if n == 0:
        active_group_mask = torch.zeros(0, dtype=torch.bool, device=device)
        return {
            "advantages": advantages,
            "active_group_mask": active_group_mask,
            "active_group_count": 0,
        }

    num_groups = int(group_ids.max().item()) + 1
    active_group_mask = torch.zeros(num_groups, dtype=torch.bool, device=device)

    for g in range(num_groups):
        mask = group_ids == g
        group_rewards = rewards[mask]

        if group_rewards.numel() == 0:
            continue

        mean = group_rewards.mean()
        # Population standard deviation (ddof=0)
        std = torch.sqrt(torch.mean((group_rewards - mean) ** 2))

        if std.item() == 0:
            # Constant-reward group: zero advantages, inactive
            advantages[mask] = torch.zeros_like(group_rewards)
            active_group_mask[g] = False
        else:
            normalized = (group_rewards - mean) / (std + epsilon)
            advantages[mask] = normalized
            active_group_mask[g] = True

    active_group_count = int(active_group_mask.sum().item())

    return {
        "advantages": advantages,
        "active_group_mask": active_group_mask,
        "active_group_count": active_group_count,
    }