import numpy as np

def bernoulli_pmf_and_moments(x, p):
    # Convert input to numpy array
    x = np.array(x)
    
    # PMF using np.where
    pmf = np.where(x == 1, p, 1 - p)
    
    # Mean and Variance
    mean = float(p)
    var = float(p * (1 - p))
    
    return pmf, mean, var