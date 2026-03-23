import numpy as np

def percentiles(x, q):
    # Convert inputs to NumPy arrays
    x = np.array(x)
    q = np.array(q)
    
    # Compute percentiles using linear interpolation
    result = np.percentile(x, q, method='linear')
    
    return result