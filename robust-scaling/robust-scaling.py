def robust_scaling(values):
    if not values:
        return []
    
    x = sorted(values)
    n = len(x)
    
    # Helper median function (safe)
    def median(arr):
        m = len(arr)
        if m == 0:
            return 0  # safety fallback
        if m % 2 == 1:
            return arr[m // 2]
        else:
            return (arr[m // 2 - 1] + arr[m // 2]) / 2
    
    # Median
    med = median(x)
    
    # Split into halves
    if n == 1:
        Q1 = Q3 = x[0]
    else:
        if n % 2 == 1:
            lower = x[:n // 2]
            upper = x[n // 2 + 1:]
        else:
            lower = x[:n // 2]
            upper = x[n // 2:]
        
        Q1 = median(lower)
        Q3 = median(upper)
    
    IQR = Q3 - Q1
    
    # Scale values
    if IQR == 0:
        return [(v - med) for v in values]
    
    return [(v - med) / IQR for v in values]