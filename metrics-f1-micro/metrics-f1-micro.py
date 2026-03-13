import numpy as np

def f1_micro(y_true, y_pred):
    # Convert to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # True Positives (correct predictions)
    tp = np.sum(y_true == y_pred)

    # Total samples
    n = len(y_true)

    # For single-label classification
    fp = n - tp
    fn = n - tp

    # Micro F1 formula
    f1 = (2 * tp) / (2 * tp + fp + fn)

    return float(f1)


# Example 1
print(round(f1_micro([0,1,1], [0,1,0]), 4))

# Example 2
print(f1_micro([0,1,2,2], [0,1,2,2]))

# Example 3
print(f1_micro([2,2,1,0], [1,2,1,0]))