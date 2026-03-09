import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    # Convert inputs to NumPy arrays
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)
    s = np.asarray(s, dtype=float)

    # Step 1: Update running squared gradient average
    s = beta * s + (1 - beta) * (g * g)

    # Step 2: Update parameters
    w = w - lr * g / (np.sqrt(s) + eps)

    return w, s


# Example 1
w = [1.0, 2.0]
g = [0.2, -0.4]
s = [0.0, 0.0]

print(rmsprop_step(w, g, s, lr=0.1))

# Example 2
w = [5.0]
g = [0.0]
s = [0.1]

print(rmsprop_step(w, g, s, lr=0.1))

# Example 3
w = [[1, 2]]
g = [[0.1, 0.2]]
s = [[0.01, 0.04]]

print(rmsprop_step(w, g, s, lr=0.1))