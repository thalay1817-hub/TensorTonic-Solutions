def moving_median(values, window_size):
    result = []
    n = len(values)

    for i in range(n - window_size + 1):
        # extract window
        window = values[i:i + window_size]
        
        # sort the window
        sorted_window = sorted(window)
        
        # find median
        if window_size % 2 == 1:  # odd
            median = sorted_window[window_size // 2]
        else:  # even
            median = (sorted_window[window_size // 2 - 1] + 
                      sorted_window[window_size // 2]) / 2
        
        result.append(median)

    return result


# Example 1
values = [1, 3, 5, 7, 9]
window_size = 3
print(moving_median(values, window_size))

# Example 2
values = [1, 2, 3, 4]
window_size = 2
print(moving_median(values, window_size))