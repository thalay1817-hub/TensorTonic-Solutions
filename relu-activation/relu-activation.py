import numpy as np

def relu(x):
    x = np.asarray(x)        # convert input to numpy array
    return np.maximum(0, x)  # apply ReLU element-wise


# Examples
print(relu([-2, -1, 0, 3]))
print(relu(5.0))
print(relu([[-1, 2], [3, -4]]))