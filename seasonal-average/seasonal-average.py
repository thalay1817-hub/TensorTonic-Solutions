def seasonal_average(series, period):
    result = []
    
    for p in range(period):
        # Collect values at indices p, p+period, p+2*period, ...
        values = [series[i] for i in range(p, len(series), period)]
        
        # Compute mean
        avg = sum(values) / len(values)
        result.append(avg)
    
    return result


# Example usage
print(seasonal_average([1, 2, 3, 4, 5, 6], 3))
print(seasonal_average([1, 2, 1, 2, 1, 2], 2))