import numpy as np

def manhattan_distance(x, y):
    x = np.array(x)
    y = np.array(y)
    
    return float(np.sum(np.abs(x - y)))