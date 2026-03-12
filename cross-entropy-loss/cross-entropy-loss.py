import numpy as np

def cross_entropy_loss(y_true, y_pred):
    # Convert inputs to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred, dtype=float)

    # Number of samples
    N = len(y_true)

    # Get predicted probabilities of the correct classes
    correct_probs = y_pred[np.arange(N), y_true]

    # Compute cross-entropy loss
    loss = -np.mean(np.log(correct_probs))

    return float(loss)


# Example 1
y_true = [0, 1]
y_pred = [[0.9, 0.1], [0.3, 0.7]]
print(round(cross_entropy_loss(y_true, y_pred), 6))

# Example 2
y_true = [2]
y_pred = [[0.1, 0.1, 0.8]]
print(round(cross_entropy_loss(y_true, y_pred), 6))

# Example 3
y_true = [1, 0, 1]
y_pred = [[0.2, 0.8], [0.6, 0.4], [0.49, 0.51]]
print(round(cross_entropy_loss(y_true, y_pred), 6))