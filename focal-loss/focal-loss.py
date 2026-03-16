import numpy as np

def focal_loss(p, y, gamma=2.0):
    # convert lists to numpy arrays
    p = np.array(p)
    y = np.array(y)

    # avoid log(0)
    p = np.clip(p, 1e-15, 1 - 1e-15)

    # focal loss terms
    term1 = (1 - p) ** gamma * y * np.log(p)
    term2 = (p ** gamma) * (1 - y) * np.log(1 - p)

    loss = -(term1 + term2)

    return np.mean(loss)


# Example
p = [0.9, 0.2, 0.7, 0.1]
y = [1, 0, 1, 0]

print(round(focal_loss(p, y, gamma=2.0), 3))