from collections import defaultdict

def bigram_probabilities(tokens):
    # Step 1: Vocabulary
    vocab = list(set(tokens))
    V = len(vocab)
    
    # Step 2: Bigram counts
    counts = defaultdict(int)
    context_counts = defaultdict(int)
    
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i + 1]
        counts[(w1, w2)] += 1
        context_counts[w1] += 1
    
    # Step 3: Compute probabilities with Add-1 smoothing
    probs = {}
    
    for w1 in vocab:
        denom = context_counts[w1] + V  # denominator
        
        for w2 in vocab:
            num = counts[(w1, w2)] + 1  # add-1 smoothing
            probs[(w1, w2)] = num / denom
    
    return counts, probs


# Example usage
tokens1 = ["a", "b", "a"]
counts1, probs1 = bigram_probabilities(tokens1)

print("Counts:", dict(counts1))
print("P(a|a):", probs1[("a","a")])
print("P(b|a):", probs1[("a","b")])

tokens2 = ["i", "love", "ml", "love", "ml"]
counts2, probs2 = bigram_probabilities(tokens2)

print("Counts:", dict(counts2))
print("P(ml|love):", probs2[("love","ml")])
print("P(love|ml):", probs2[("ml","love")])