def top_k_recommendations(scores, rated_indices, k):
    # Step 1: Filter unrated items and create (score, index) pairs
    candidates = [(scores[i], i) for i in range(len(scores)) if i not in rated_indices]

    # Step 2: Sort by score in descending order
    candidates.sort(key=lambda x: -x[0])

    # Step 3: Return indices of top-k items
    result = [idx for (_, idx) in candidates[:k]]

    return result


# Example 1
scores = [3.5, 1.2, 4.8, 2.1, 5.0]
rated_indices = {0, 2}
k = 2
print(top_k_recommendations(scores, rated_indices, k))

# Example 2
scores = [1.0, 3.0, 2.0]
rated_indices = {}
k = 2
print(top_k_recommendations(scores, rated_indices, k))