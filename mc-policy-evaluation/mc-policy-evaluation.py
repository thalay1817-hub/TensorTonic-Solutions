import numpy as np

def mc_policy_evaluation(episodes, gamma, n_states):
    returns_sum = np.zeros(n_states)
    returns_count = np.zeros(n_states)

    for episode in episodes:
        G = 0
        visited = set()

        # Process episode backward to compute returns
        for t in reversed(range(len(episode))):
            state, reward = episode[t]
            G = reward + gamma * G

            # Check if this is the first visit of the state
            if state not in [s for s, _ in episode[:t]]:
                returns_sum[state] += G
                returns_count[state] += 1

    # Compute value function
    V = np.zeros(n_states)
    for s in range(n_states):
        if returns_count[s] > 0:
            V[s] = returns_sum[s] / returns_count[s]

    return V


# Example 1
episodes = [[(0,1),(1,2),(2,3)]]
gamma = 1
n_states = 3
print(mc_policy_evaluation(episodes, gamma, n_states))

# Example 2
episodes = [[(0,1),(0,-5),(0,2)]]
gamma = 1
n_states = 1
print(mc_policy_evaluation(episodes, gamma, n_states))