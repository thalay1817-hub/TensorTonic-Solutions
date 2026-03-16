def gae(rewards, values, gamma, lam):
    T = len(rewards)
    advantages = [0.0] * T
    last_adv = 0.0

    for t in range(T - 1, -1, -1):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        advantages[t] = delta + gamma * lam * last_adv
        last_adv = advantages[t]

    return advantages


# Example 1
rewards = [1, 1, 1]
values = [0, 0, 0, 0]
gamma = 1.0
lam = 1.0

print(gae(rewards, values, gamma, lam))


# Example 2
rewards = [1, 0, 5]
values = [1, 2, 3, 0]
gamma = 0.9
lam = 0.95

print(gae(rewards, values, gamma, lam))