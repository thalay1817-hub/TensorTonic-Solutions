def double_exponential_smoothing(series, alpha, beta):
    n = len(series)
    
    # Initialize level and trend
    level = series[0]
    trend = series[1] - series[0] if n > 1 else 0.0
    
    result = [float(level)]
    
    for t in range(1, n):
        prev_level = level
        
        # Update level
        level = alpha * series[t] + (1 - alpha) * (level + trend)
        
        # Update trend
        trend = beta * (level - prev_level) + (1 - beta) * trend
        
        result.append(float(level))
    
    return result


# Example usage
print(double_exponential_smoothing([10, 20, 30], 0.5, 0.5))
print(double_exponential_smoothing([5, 5, 5, 5], 0.9, 0.1))