import math

def perplexity(prob_distributions, actual_tokens):
    N = len(actual_tokens)
    
    log_sum = 0.0
    
    for i in range(N):
        p = prob_distributions[i][actual_tokens[i]]
        
        # Avoid log(0)
        if p <= 0:
            return float('inf')
        
        log_sum += math.log(p)
    
    cross_entropy = -log_sum / N
    perplexity = math.exp(cross_entropy)
    
    return perplexity