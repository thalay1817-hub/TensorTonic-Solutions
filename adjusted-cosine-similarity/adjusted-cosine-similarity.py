import math

def adjusted_cosine_similarity(ratings, item_i, item_j):
    
    numerator = 0
    sum_i = 0
    sum_j = 0
    
    for user in ratings:
        # Compute user's mean rating (ignore zeros)
        rated = [r for r in user if r != 0]
        if not rated:
            continue
        mean_u = sum(rated) / len(rated)
        
        ri = user[item_i]
        rj = user[item_j]
        
        # Only consider users who rated both items
        if ri != 0 and rj != 0:
            ci = ri - mean_u
            cj = rj - mean_u
            
            numerator += ci * cj
            sum_i += ci ** 2
            sum_j += cj ** 2
    
    denom = math.sqrt(sum_i) * math.sqrt(sum_j)
    
    if denom == 0:
        return 0.0
    
    return numerator / denom


# Example 1
ratings = [[5,3,0],[4,0,2],[0,1,4]]
print(adjusted_cosine_similarity(ratings, 0, 1))

# Example 2
ratings = [[5,1],[4,2],[3,3]]
print(adjusted_cosine_similarity(ratings, 0, 1))