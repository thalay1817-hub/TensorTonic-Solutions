def differencing(series, order):
    result = list(series)  # make a copy
    
    for _ in range(order):
        result = [result[i] - result[i-1] for i in range(1, len(result))]
    
    return result


# Example 1
series = [1, 3, 6, 10, 15]