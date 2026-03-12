def rating_normalization(matrix):
    result = []

    for row in matrix:
        # Get non-zero ratings
        rated = [v for v in row if v != 0]

        # If user has no ratings
        if len(rated) == 0:
            result.append([0.0]*len(row))
            continue

        # Compute mean of rated items
        mean_rating = sum(rated) / len(rated)

        # Normalize row
        normalized = [(v - mean_rating) if v != 0 else 0.0 for v in row]

        result.append(normalized)

    return result


# Example 1
matrix = [[5, 3, 0, 1],
          [4, 0, 0, 1],
          [1, 1, 0, 5]]

print([[round(x,3) for x in row] for row in rating_normalization(matrix)])


# Example 2
matrix = [[2, 4, 6]]
print(rating_normalization(matrix))