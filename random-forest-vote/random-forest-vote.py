import numpy as np

def random_forest_vote(predictions):
    predictions = np.asarray(predictions)
    
    T, N = predictions.shape
    result = []

    for i in range(N):  # loop over samples
        votes = {}
        
        # count votes from all trees
        for t in range(T):
            label = predictions[t, i]
            votes[label] = votes.get(label, 0) + 1
        
        # find maximum vote count
        max_count = max(votes.values())
        
        # collect labels with max votes
        candidates = [label for label, count in votes.items() if count == max_count]
        
        # tie-breaking: choose smallest label
        result.append(min(candidates))
    
    return result


# Example 1
predictions = [[0, 1, 0],
               [0, 1, 1],
               [0, 0, 0]]

print(random_forest_vote(predictions))


# Example 2
predictions = [[0, 1],
               [1, 0]]

print(random_forest_vote(predictions))