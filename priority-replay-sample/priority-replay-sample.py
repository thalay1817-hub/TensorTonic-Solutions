def priority_replay_sample(priorities, alpha, beta):
    n = len(priorities)
    
    # Step 1: powered priorities
    powered = [p ** alpha for p in priorities]
    
    # Step 2: probabilities
    total = sum(powered)
    probs = [p / total for p in powered]
    
    # Step 3: importance sampling weights
    weights = [(n * p) ** (-beta) for p in probs]
    
    # Step 4: normalize weights
    max_w = max(weights)
    weights = [w / max_w for w in weights]
    
    return [probs, weights]