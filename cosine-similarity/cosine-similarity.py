import numpy as np

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    # Compute norms
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    # Handle zero vectors
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # Cosine similarity
    similarity = np.dot(a, b) / (norm_a * norm_b)
    return float(similarity)


# Example 1
print(cosine_similarity([1,2,3], [2,4,6]))

# Example 2
print(cosine_similarity([1,0], [0,1]))