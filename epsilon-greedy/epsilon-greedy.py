import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    q_values = np.asarray(q_values)
    n_actions = len(q_values)

    # Random generator
    if rng is None:
        rng = np.random

    # Exploration
    if rng.random() < epsilon:
        if hasattr(rng, "integers"):
            return rng.integers(n_actions)
        else:
            return rng.randint(n_actions)

    # Exploitation (greedy action)
    return int(np.argmax(q_values))


# Examples
print(epsilon_greedy([1,2,0.5], 0))  # greedy → 1
print(epsilon_greedy([1,2,0.5], 1))  # random → 0,1, or 2
