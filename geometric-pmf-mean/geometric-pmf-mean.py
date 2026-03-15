import numpy as np

def geometric_pmf_mean(k, p):
    # Convert k to numpy array
    k = np.array(k)
    
    # Calculate PMF using formula
    pmf = (1 - p) ** (k - 1) * p
    
    # Mean of geometric distribution
    mean = 1 / p
    
    return pmf, mean


# Example
k = [1, 2, 3]
p = 0.5

result = geometric_pmf_mean(k, p)
print(result)