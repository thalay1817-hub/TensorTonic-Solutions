import numpy as np

def generator(z, W, b):
    return np.round(np.tanh(np.dot(z, W) + b), 4)