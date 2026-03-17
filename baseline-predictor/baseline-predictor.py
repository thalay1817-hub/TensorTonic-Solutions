def baseline_predict(matrix, pairs):
    n_users = len(matrix)
    n_items = len(matrix[0])
    
    # Step 1: Global mean (mu)
    ratings = [val for row in matrix for val in row if val != 0]
    mu = sum(ratings) / len(ratings) if ratings else 0.0
    
    # Step 2: User biases
    user_bias = []
    for u in range(n_users):
        user_ratings = [matrix[u][i] for i in range(n_items) if matrix[u][i] != 0]
        if user_ratings:
            user_mean = sum(user_ratings) / len(user_ratings)
            user_bias.append(user_mean - mu)
        else:
            user_bias.append(0.0)
    
    # Step 3: Item biases
    item_bias = []
    for i in range(n_items):
        item_ratings = [matrix[u][i] for u in range(n_users) if matrix[u][i] != 0]
        if item_ratings:
            item_mean = sum(item_ratings) / len(item_ratings)
            item_bias.append(item_mean - mu)
        else:
            item_bias.append(0.0)
    
    # Step 4: Predictions (NO rounding)
    results = []
    for u, i in pairs:
        pred = mu + user_bias[u] + item_bias[i]
        results.append(pred)
    
    return results


# Example
print(baseline_predict(
    [[5, 3, 0], [4, 0, 1], [0, 1, 5]],
    [[0, 2], [1, 1], [2, 0]]
))