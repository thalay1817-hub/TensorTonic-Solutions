def min_max_scaling(data):
    rows = len(data)
    cols = len(data[0])

    # create result matrix filled with 0.0
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]

    for j in range(cols):
        # extract column
        column = [data[i][j] for i in range(rows)]

        min_val = min(column)
        max_val = max(column)
        range_val = max_val - min_val

        for i in range(rows):
            if range_val == 0:
                result[i][j] = 0.0
            else:
                result[i][j] = (data[i][j] - min_val) / range_val

    return result


# Example 1
data = [[1, 10], [2, 20], [3, 30]]
print(min_max_scaling(data))

# Example 2
data = [[0, 0], [10, 100], [20, 50]]
print(min_max_scaling(data))