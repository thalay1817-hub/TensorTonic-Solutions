import math

def novelty_score(recommendations, item_counts, n_users):
    # Edge case: empty recommendations
    if not recommendations:
        return 0.0
    
    total_novelty = 0.0
    
    for item in recommendations:
        count = item_counts[item]
        
        # Avoid division by zero or log(0)
        if count == 0 or n_users == 0:
            continue
        
        popularity = count / n_users
        total_novelty += -math.log2(popularity)
    
    # Average novelty
    return total_novelty / len(recommendations)


# Example usage
print(round(novelty_score([0, 1], [100, 100], 100), 4))
print(round(novelty_score([0, 1], [1, 1], 100), 4))