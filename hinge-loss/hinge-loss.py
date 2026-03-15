import numpy as np

def hinge_loss(y_true, y_score, margin=1, reduction="mean"):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Vectorized hinge loss
    loss = np.maximum(0, margin - y_true * y_score)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


# Example 1
y_true = [1, 1, -1]
y_score = [2, 0, 0]

print(hinge_loss(y_true, y_score))
# Output: 0.66666667


# Example 2
y_true = [-1, 1]
y_score = [-3, 0.5]

print(hinge_loss(y_true, y_score))
# Output: 0.25