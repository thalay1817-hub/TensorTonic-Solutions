import numpy as np

def mean_squared_error(y_pred, y_true):
    # convert lists to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # compute squared differences and take mean
    mse = np.mean((y_pred - y_true) ** 2)

    return mse


# Example 1
y_pred = [2, 3]
y_true = [1, 1]
print(mean_squared_error(y_pred, y_true))

# Example 2
y_pred = [0, 0, 0]
y_true = [0, 0, 0]
print(mean_squared_error(y_pred, y_true))