def simple_moving_average(values, window_size):
    n = len(values)
    result = []

    for i in range(n - window_size + 1):
        window = values[i:i + window_size]
        avg = sum(window) / window_size
        result.append(float(avg))

    return result