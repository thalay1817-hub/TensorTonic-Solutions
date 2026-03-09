import numpy as np

def precision_recall_at_k(recommended, relevant, k):
    # Top-k recommended items
    top_k = recommended[:k]

    # Convert relevant items to set for fast lookup
    relevant_set = set(relevant)

    # Count hits
    hits = sum(1 for item in top_k if item in relevant_set)

    # Compute metrics
    precision = hits / k if k > 0 else 0
    recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0

    return [precision, recall]


# Example 1
recommended = [1, 3, 5, 7, 9]
relevant = [1, 2, 3, 4, 5]
k = 3

print(precision_recall_at_k(recommended, relevant, k))


# Example 2
recommended = [10, 20, 30]
relevant = [1, 2, 3]
k = 3

print(precision_recall_at_k(recommended, relevant, k))