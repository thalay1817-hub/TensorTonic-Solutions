import numpy as np

def sigmoid(x):
    
    x = np.asarray(x, dtype=float)
    return 1 / (1 + np.exp(-x))
print(sigmoid(0))              
print(sigmoid([1, 2, 3]))     
print(sigmoid(np.array([-1, 0, 1])))  
