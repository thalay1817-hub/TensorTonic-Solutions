def policy_gradient_loss(log_probs, rewards, gamma):
    T = len(rewards)

    # Step 1: Compute discounted returns G
    G = [0] * T
    G[-1] = rewards[-1]

    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]

    # Step 2: Compute mean return (baseline)
    mean_G = sum(G) / T

    # Step 3: Compute advantages
    advantages = [g - mean_G for g in G]

    # Step 4: Compute policy gradient loss
    loss = -sum(lp * a for lp, a in zip(log_probs, advantages)) / T

    return loss


# Example 1
log_probs = [-1.0, -2.0, -0.5]
rewards = [1, 1, 1]
gamma = 1.0
print(round(policy_gradient_loss(log_probs, rewards, gamma), 4))


# Example 2
log_probs = [-1.0, -0.5]
rewards = [0, 10]
gamma = 0.0
print(round(policy_gradient_loss(log_probs, rewards, gamma), 2))