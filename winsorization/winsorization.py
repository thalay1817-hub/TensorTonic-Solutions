import math

def percentile(arr, p):
    # Sort array
    arr_sorted = sorted(arr)
    n = len(arr_sorted)
    
    # Compute index
    k = (n - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    
    # Interpolation
    if f == c:
        return arr_sorted[int(k)]
    
    return arr_sorted[f] + (k - f) * (arr_sorted[c] - arr_sorted[f])


def winsorize(values, lower_pct, upper_pct):
    # Compute bounds
    lower = percentile(values, lower_pct)
    upper = percentile(values, upper_pct)
    
    # Clip values (preserve original order)
    result = [max(lower, min(upper, v)) for v in values]
    
    return result


# Example usage
print(winsorize([1,2,3,4,5,6,7,8,9,10], 10, 90))
print(winsorize([1,2,3,4,5], 0, 100))