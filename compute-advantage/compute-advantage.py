import numpy as np

def compute_advantage(states, rewards, V, gamma):
    T = len(rewards)
    G = [0.0] * T
    
    # Step 1: Compute returns (backward)
    G[T - 1] = rewards[T - 1]
    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]
    
    # Step 2: Compute advantage
    A = [G[t] - V[states[t]] for t in range(T)]
    
    return A


# Example usage
print(compute_advantage([0,1,2], [1,2,3], np.array([0.5,1,1.5]), 1))
print(compute_advantage([0,1,2], [1,2,3], np.array([0,0,0]), 0))