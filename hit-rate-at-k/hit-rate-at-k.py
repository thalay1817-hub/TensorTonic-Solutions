def hit_rate_at_k(recommendations, ground_truth, k):
    hits = 0
    n_users = len(recommendations)

    for recs, truth in zip(recommendations, ground_truth):
        # Take top-K recommendations
        top_k = set(recs[:k])
        truth_set = set(truth)

        # Check if intersection is not empty
        if top_k & truth_set:
            hits += 1

    return hits / n_users