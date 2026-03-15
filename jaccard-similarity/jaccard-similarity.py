def jaccard_similarity(set_a, set_b):
    A = set(set_a)
    B = set(set_b)
    
    union = A | B
    intersection = A & B
    
    # handle empty union
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)