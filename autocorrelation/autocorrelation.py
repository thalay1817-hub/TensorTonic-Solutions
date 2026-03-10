def autocorrelation(series, max_lag):
    n = len(series)
    
    # Mean of the series
    mean = sum(series) / n
    
    # Total variance γ0
    variance = sum((x - mean) ** 2 for x in series)
    
    # Edge case: constant series
    if variance == 0:
        return [1.0] + [0.0] * max_lag
    
    result = []
    
    # Loop over lags
    for k in range(max_lag + 1):
        cov = 0
        
        # Cross-product sum
        for t in range(n - k):
            cov += (series[t] - mean) * (series[t + k] - mean)
        
        r_k = cov / variance
        result.append(r_k)
    
    return result


# Example 1
series = [1, 2, 3, 4, 5]
print(autocorrelation(series, 2))

# Example 2
series = [1, -1, 1, -1, 1, -1]
print(autocorrelation(series, 2))