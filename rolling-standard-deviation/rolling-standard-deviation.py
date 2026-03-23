import math

def rolling_std(values, window_size):
    n = len(values)
    
    if window_size <= 0 or window_size > n:
        return []
    
    result = []
    
    for i in range(n - window_size + 1):
        window = values[i:i + window_size]
        
        # Mean
        mean = sum(window) / window_size
        
        # Population variance
        var = sum((x - mean) ** 2 for x in window) / window_size
        
        # Standard deviation
        std = math.sqrt(var)
        
        result.append(std)
    
    return result