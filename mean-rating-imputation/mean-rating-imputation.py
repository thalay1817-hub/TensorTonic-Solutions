import numpy as np

def mean_rating_imputation(ratings, mode="user"):
    ratings = np.array(ratings, dtype=float)
    result = ratings.copy()

    if mode == "user":
        # Iterate over each row (user)
        for i in range(result.shape[0]):
            row = result[i]
            non_zero = row[row != 0]

            if len(non_zero) == 0:
                continue  # skip if no ratings

            mean_val = np.mean(non_zero)
            row[row == 0] = mean_val

    elif mode == "item":
        # Iterate over each column (item)
        for j in range(result.shape[1]):
            col = result[:, j]
            non_zero = col[col != 0]

            if len(non_zero) == 0:
                continue  # skip if no ratings

            mean_val = np.mean(non_zero)
            col[col == 0] = mean_val

    else:
        raise ValueError("mode must be 'user' or 'item'")

    return result.tolist()