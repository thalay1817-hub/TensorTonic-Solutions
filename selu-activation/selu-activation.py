import numpy as np
import math

def selu(x):
    # constants
    lam = 1.0507
    alpha = 1.6733
    
    x = np.asarray(x, dtype=float)
    
    # apply SELU element-wise
    return np.where(x > 0,
                    lam * x,
                    lam * alpha * (np.exp(x) - 1))


# Examples
print(np.round(selu([1.0, -1.0, 0.0]), 4))
print(np.round(selu([0.5, 1.5, 2.5]), 4))
