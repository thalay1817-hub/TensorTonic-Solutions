import numpy as np
from collections import Counter

def mean_median_mode(x):
    
    x = np.asarray(x, dtype=float)

    # Mean
    mean_val = float(np.mean(x))

    # Median
    median_val = float(np.median(x))

    # Mode
    freq = Counter(x)
    max_count = max(freq.values())
    modes = [k for k, v in freq.items() if v == max_count]
    mode_val = float(min(modes))   # choose smallest if multiple modes

    return mean_val, median_val, mode_val


# Examples
print(mean_median_mode([1,1,2,3]))
print(mean_median_mode([5]))
print(mean_median_mode([1,2,2,3,3]))