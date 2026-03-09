import numpy as np

def entropy_node(y):
    y = np.asarray(y)

    # Get class counts
    _, counts = np.unique(y, return_counts=True)

    # Convert to probabilities
    p = counts / counts.sum()

    # Remove zero probabilities (for numerical stability)
    p = p[p > 0]

    # Compute entropy
    entropy = -np.sum(p * np.log2(p))

    return entropy


# Example 1
y = [1,1,1,1]
print(entropy_node(y))

# Example 2
y = [0,1,0,1]
print(entropy_node(y))