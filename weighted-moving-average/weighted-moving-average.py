def weighted_moving_average(values, weights):
    n = len(values)
    k = len(weights)
    
    w_sum = sum(weights)
    result = []
    
    # Slide window
    for i in range(n - k + 1):
        weighted_sum = sum(weights[j] * values[i + j] for j in range(k))
        result.append(weighted_sum / w_sum)
    
    return result


# Example 1
values = [1, 2, 3, 4, 5]
weights = [1, 1, 1]
print(weighted_moving_average(values, weights))

# Example 2
values = [10, 20, 30, 40]
weights = [1, 2, 3]
print([round(x, 3) for x in weighted_moving_average(values, weights)])