def user_based_cf_prediction(similarities, ratings):
    num = 0.0
    den = 0.0

    # Iterate through similarities and ratings
    for sim, rating in zip(similarities, ratings):
        if sim > 0:   # consider only positive similarities
            num += sim * rating
            den += sim

    # If no positive similarities
    if den == 0:
        return 0.0

    return num / den


# Example 1
similarities = [0.9, 0.8, 0.3]
ratings = [4, 5, 2]
print(round(user_based_cf_prediction(similarities, ratings), 3))

# Example 2
similarities = [0.8, -0.2, 0.6]
ratings = [5, 1, 3]
print(round(user_based_cf_prediction(similarities, ratings), 3))