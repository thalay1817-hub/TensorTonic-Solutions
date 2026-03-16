def rank_transform(values):
    n = len(values)
    ranks = [0] * n

    # sort indices based on values
    sorted_indices = sorted(range(n), key=lambda i: values[i])

    i = 0
    while i < n:
        j = i
        
        # find group of equal values (ties)
        while j + 1 < n and values[sorted_indices[j]] == values[sorted_indices[j + 1]]:
            j += 1
        
        # ranks are 1-based
        avg_rank = (i + 1 + j + 1) / 2
        
        # assign same average rank to tied values
        for k in range(i, j + 1):
            ranks[sorted_indices[k]] = avg_rank
        
        i = j + 1

    return ranks


# Example 1
values = [10, 30, 20]
print(rank_transform(values))

# Example 2
values = [1, 2, 2, 3]
print(rank_transform(values))