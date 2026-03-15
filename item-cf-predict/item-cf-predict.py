def item_cf_predict(user_ratings, item_similarities, target):
    numerator = 0.0
    denominator = 0.0

    for i in range(len(user_ratings)):
        if i == target:
            continue

        rating = user_ratings[i]
        sim = item_similarities[i]

        if rating == 0 or sim <= 0:
            continue

        numerator += sim * rating
        denominator += sim

    if denominator == 0:
        return 0.0

    return numerator / denominator