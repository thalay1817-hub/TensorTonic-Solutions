import numpy as np

def sample_var_std(x):
    x = np.array(x)
    n = len(x)
    
    if n < 2:
        raise ValueError("At least 2 data points are required")
    
    mean_x = np.mean(x)
    
    # Sample variance (Bessel's correction)
    var = np.sum((x - mean_x) ** 2) / (n - 1)
    
    # Standard deviation
    std = np.sqrt(var)
    
    return var, std


# Examples
print(sample_var_std([1, 2, 3]))   # Output: (1.0, 1.0)
print(sample_var_std([5, 7]))      # Output: (2.0, 1.414...)
print(sample_var_std([4, 4, 4, 4]))# Output: (0.0, 0.0)