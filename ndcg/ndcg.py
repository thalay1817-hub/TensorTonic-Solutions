import numpy as np

def ndcg(relevance_scores, k):
    rel = np.array(relevance_scores, dtype=float)

    # Use only top-k (or all if k too large)
    k = min(k, len(rel))
    rel_k = rel[:k]

    # Positions (1-based index)
    positions = np.arange(1, k + 1)

    # DCG
    dcg = np.sum((2**rel_k - 1) / np.log2(positions + 1))

    # Ideal DCG (sorted descending)
    ideal_rel = np.sort(rel)[::-1][:k]
    idcg = np.sum((2**ideal_rel - 1) / np.log2(positions + 1))

    # Handle edge case
    if idcg == 0:
        return 0.0

    return float(dcg / idcg)