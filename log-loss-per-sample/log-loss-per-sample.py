import math

def log_loss(y_true, y_pred, eps=1e-15):
    losses = []

    for y, p in zip(y_true, y_pred):
        # Clip probabilities
        p_hat = min(max(p, eps), 1 - eps)

        # Compute log loss
        loss = -(y * math.log(p_hat) + (1 - y) * math.log(1 - p_hat))
        losses.append(loss)

    return losses