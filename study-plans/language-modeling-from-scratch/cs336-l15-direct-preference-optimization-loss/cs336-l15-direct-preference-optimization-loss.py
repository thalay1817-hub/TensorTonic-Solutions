import torch
import torch.nn.functional as F

def direct_preference_optimization_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta, label_smoothing):
    dtype = policy_chosen.dtype
    device = policy_chosen.device

    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected

    logits = beta * (policy_margin - reference_margin)

    per_example_loss = -(1 - label_smoothing) * F.logsigmoid(logits) - label_smoothing * F.logsigmoid(-logits)

    loss = per_example_loss.mean()

    chosen_rewards = beta * (policy_chosen - reference_chosen)
    rejected_rewards = beta * (policy_rejected - reference_rejected)

    preference_accuracy = (chosen_rewards > rejected_rewards).to(dtype).mean()

    return {
        "loss": loss.to(dtype),
        "per_example_loss": per_example_loss.to(dtype),
        "chosen_rewards": chosen_rewards.to(dtype),
        "rejected_rewards": rejected_rewards.to(dtype),
        "preference_accuracy": preference_accuracy.to(dtype),
    }