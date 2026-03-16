def frequency_encoding(values):
    n = len(values)
    
    # count occurrences
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    
    # convert each value to its frequency
    result = [counts[v] / n for v in values]
    
    return result


# Example 1
values = ["a", "b", "a", "c", "a"]
print(frequency_encoding(values))

# Example 2
values = ["cat", "dog", "cat", "cat", "dog"]
print(frequency_encoding(values))