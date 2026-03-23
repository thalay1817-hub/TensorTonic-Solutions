import math
from collections import Counter

def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def bleu_score(candidate, reference, max_n):
    c_len = len(candidate)
    r_len = len(reference)
    
    precisions = []
    
    for n in range(1, max_n + 1):
        # Extract n-grams
        cand_ngrams = get_ngrams(candidate, n)
        ref_ngrams = get_ngrams(reference, n)
        
        # Count frequencies
        cand_counts = Counter(cand_ngrams)
        ref_counts = Counter(ref_ngrams)
        
        # Clipped counts
        clipped = 0
        total = sum(cand_counts.values())
        
        for ng in cand_counts:
            clipped += min(cand_counts[ng], ref_counts.get(ng, 0))
        
        # Precision
        if total == 0:
            precisions.append(0)
        else:
            precisions.append(clipped / total)
    
    # If any precision is zero → BLEU = 0
    if any(p == 0 for p in precisions):
        return 0.0
    
    # Brevity Penalty (BP)
    if c_len >= r_len:
        BP = 1.0
    else:
        BP = math.exp(1 - r_len / c_len)
    
    # Geometric mean of precisions
    log_sum = sum(math.log(p) for p in precisions)
    bleu = BP * math.exp(log_sum / max_n)
    
    return float(bleu)