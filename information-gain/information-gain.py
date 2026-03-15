import numpy as np

def entropy(y):
    if len(y) == 0:
        return 0.0
    
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    
    return -np.sum(probs * np.log2(probs))


def information_gain(y, mask):
    y = np.array(y)
    mask = np.array(mask)

    y_left = y[mask]
    y_right = y[~mask]

    n = len(y)
    n_left = len(y_left)
    n_right = len(y_right)

    # If one side empty → no information
    if n_left == 0 or n_right == 0:
        return 0.0

    H_parent = entropy(y)
    H_left = entropy(y_left)
    H_right = entropy(y_right)

    IG = H_parent - ((n_left/n) * H_left + (n_right/n) * H_right)

    return IG