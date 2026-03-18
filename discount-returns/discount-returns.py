def discount_returns(rewards, gamma):
    T = len(rewards)
    G = [0.0] * T
    
    # Base case
    G[T - 1] = rewards[T - 1]
    
    # Compute backwards
    for t in range(T - 2, -1, -1):
        G[t] = rewards[t] + gamma * G[t + 1]
    
    return G


# Example usage
print(discount_returns([1, 1, 1], 1.0))
print(discount_returns([0, 0, 0, 10], 0.9))