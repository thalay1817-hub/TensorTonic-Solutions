def target_encoding(categories, targets):
    # Step 1: compute sums and counts
    sums = {}
    counts = {}
    
    for cat, target in zip(categories, targets):
        sums[cat] = sums.get(cat, 0) + target
        counts[cat] = counts.get(cat, 0) + 1
    
    # Step 2: compute means
    means = {cat: sums[cat] / counts[cat] for cat in sums}
    
    # Step 3: map categories to means
    return [means[cat] for cat in categories]